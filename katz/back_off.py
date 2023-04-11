from katz.good_turing import GoodTuring


class BackOff:

    def __init__(self, corpus):
        """Class to create a Katz back-off model from a corpus of text and evaluate
        the probability of a given tuple of words based on this corpus
        
        Args:
            :corpus (list of tuples): Tuples of words which form the corpus
            
        Returns:
            Backoff: Katz back-off model based on this corpus
        """
    
        all_len = [len(s) for s in corpus]
         
        # Create the required GoodTuring Objects
        self.all_gt = [None] * (max(all_len)+1)
        for i in range(1, max(all_len)+1):
#            d = [s[:i] for s in corpus]
#            # Deal with edge effect
#            for j in range(all_len[0] - i):
#                d = d + [corpus[-1][j+1:i+j+1]]
#            self.all_gt[i] = GoodTuring(d)
            
            d = [s[-i:] for s in corpus if len(s) >= i]   # The final n terms of the tuples
            d_start = [dd[:-1] for dd in d] # The first n-1 terms of the n-tuple

            # These come in pairs to you can evaluate:
            #       C(w_{i-n+1) ... w_i)        [first item]
            #       C(w_{i-n+1) ... w_{i-1})    [second item]
            # Each pair in list is for a different n
            if i == 1:
                self.all_gt[i] = [GoodTuring(d), None]
            else:
                self.all_gt[i] = [GoodTuring(d), GoodTuring(d_start)]
        
        # Find all unique words in the corpus
        self.words = list(sum(corpus, ()))
        self.words = list(sorted(set(self.words), key=self.words.index))
        
    def get_d(self, phrase):
        """
        Compute the amount of discounting found by Good-Turing estimation
        
        Args:
            :phrase (tuple): Collection of words to find discounting for
            
        Returns:
            :d (float): Good-Turing estimate for discounting
            
        """
        cstar = self.all_gt[len(phrase)][0].expected_count(phrase)  # C^*(w_{i-n+1} ... w_i)
        c = self.all_gt[len(phrase)][0].actual_count(phrase)        # C(w_{i-n+1} ... w_i)
        return cstar / c
        
    def sort_endings(self, phrase):
        """
        Find all ways of completing a phrase such that the new phrase appears
        in the corpus
        
        Args:
            :phrase (tuple): Collection of words to find valid completions to
            
        Returns:
            :seen (list): List of words which can complete the phrase to produce a phrase found in the corpus
            :unseen (list): List of words which, if appended to the phrase, produce a phrase NOT found in the corpus
        """
        seen = []
        unseen = []
        for i, w in enumerate(self.words):
            if self.all_gt[len(phrase)+1][0].actual_count(phrase + (w,)) == 0:
                unseen.append(w)
            else:
                seen.append(w)
                
        return seen, unseen
        
    def get_alpha(self, old_phrase):
        """
        Compute the back-off weight
        
        Args:
            :old_phrase (tuple): The (n-1)-length tuple used to find the back-off weight for the n-length tuple
            
        Returns:
            :alpha (float): The back-off weight
            :beta (float): The left-over probability mass for the (n-1)-gram
        """
        seen, unseen = self.sort_endings(old_phrase)
    
        beta = 0.
        # Sum over w_i: C(w_{i-n+1} ... w_i) > 0 [i.e. seen]
        for w in seen:
            new_phrase = old_phrase + (w,)
            d = self.get_d(new_phrase)      # d_{w_{i-n+1} ... w_i}
            cnew = self.all_gt[len(new_phrase)][0].actual_count(new_phrase) # C(w_{i-n+1} ... w_i)
            beta += d * cnew
        cold = self.all_gt[len(new_phrase)][1].actual_count(old_phrase)  # [1] since want to get  C(w_{i-n+1} ... w_{i-1})
        beta = 1 - beta / cold  # beta_{w_{i-n+1} ... w_{i-1}}
        
        # Expect len(seen) < len(unseen)
        # Economical to only run len(seen) times and then use normalisation of probs
        alpha = 1.
        # Sum over w_i: C(w_{i-n+1} ... w_i) > 0 [i.e. seen]
        for w in seen:
            alpha -= self.get_pbo(w, old_phrase[1:])       # Pbo(w_i | w_{i-n+2} ... w_{i-1})
        alpha = beta / alpha    # alpha_{w_{i-n+1} ... w_{i-1}}
        
        return alpha, beta
        
    def get_pbo(self, wnew, old_phrase):
        """
        Compute the probability for word wnew given the preceeding set of words old_phrase
        
        Args:
            :wnew (str): The new word we wish to know the probability of obtaining
            :old_phrase (tuple): The preceeding phrase
            
        Returns:
            :pbo (float): The conditional probability P(wnew|old_phrase)
        
        """
    
        new_phrase = old_phrase + (wnew,)
        cnew = self.all_gt[len(new_phrase)][0].actual_count(new_phrase) # C(w_{i-n+1} ... w_i)
        
        if len(old_phrase) == 0:
            pbo = self.all_gt[1][0].actual_count((wnew,)) / len(self.all_gt[1][0].corpus)
            
        elif cnew > 0:
            d = self.get_d(new_phrase)  # d_{w_{i-n+1} ... w_i}
            cold = self.all_gt[len(new_phrase)][1].actual_count(old_phrase)     # C(w_{i-n+1} ... w_{i-1})
            pbo = d * cnew / cold
            
        elif len(old_phrase) > 1: # Never saw new phrase, and old phrase has length > 1
            if self.all_gt[len(new_phrase)][1].actual_count(old_phrase) > 0:
                # old_phrase appears as the start somewhere
                alpha, beta = self.get_alpha(old_phrase)            # alpha_{w_{i-n+1} ... w_{i-1}}
                pbo = alpha * self.get_pbo(wnew, old_phrase[1:])    # alpha_{w_{i-n+1} ... w_{i-1}} * Pbo(w_i | w_{i-n+2} ... w_{i-1})
            else:
                # If no data for (n-1)-gram, skip n-1 and use n-2
                pbo = self.get_pbo(wnew, old_phrase[1:])            # Pbo(w_i | w_{i-n+2} ... w_{i-1})
                
        elif self.all_gt[len(new_phrase)][1].actual_count(old_phrase) > 0:
            # Never saw new phrase, but have an old phrase of length 1 which appears somehwere in corpus
            alpha, beta = self.get_alpha(old_phrase)                # alpha_{w_{i-1}}
            pbo = alpha * self.all_gt[1][0].actual_count((wnew,)) / len(self.all_gt[1][0].corpus)   # alpha_{w_{i-1}} * Pbo(w_{i} | . )

        else:
            pbo = self.all_gt[1][0].actual_count((wnew,)) / len(self.all_gt[1][0].corpus)

        return pbo
