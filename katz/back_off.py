from .good_turing import GoodTuring


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
        if len(set(all_len)) != 1:
            raise Exception("Not all items in corpus have the same length")
         
        # Create the required GoodTuring Objects
        self.all_gt = [None] * (all_len[0]+1)
        for i in range(1, all_len[0]+1):
            d = [s[:i] for s in corpus]
            # Deal with edge effect
            for j in range(all_len[0] - i):
                d = d + [corpus[-1][j+1:i+j+1]]
            
            self.all_gt[i] = GoodTuring(d)
        
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
        cstar = self.all_gt[len(phrase)].expected_count(phrase)
        c = self.all_gt[len(phrase)].actual_count(phrase)
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
            if self.all_gt[len(phrase)+1].actual_count(phrase + (w,)) == 0:
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
        for w in seen:
            new_phrase = old_phrase + (w,)
            d = self.get_d(new_phrase)
            cnew = self.all_gt[len(new_phrase)].actual_count(new_phrase)
            beta += d * cnew
            
        cold = self.all_gt[len(old_phrase)].actual_count(old_phrase)
        beta = 1 - beta / cold
        
        # Expect len(seen) < len(unseen)
        # Economical to only run len(seen) times and then use normalisation of probs
        alpha = 1.
        for w in seen:
            if len(old_phrase) > 1:
                alpha -= self.get_pbo(w, old_phrase[1:])
            else:
                alpha -= self.all_gt[1].actual_count((w,)) / len(self.all_gt[1].corpus)
        alpha = beta / alpha
        
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
        cnew = self.all_gt[len(new_phrase)].actual_count(new_phrase)
        if cnew > 0:
            d = self.get_d(new_phrase)
            cold = self.all_gt[len(old_phrase)].actual_count(old_phrase)
            pbo = d * cnew / cold
        elif len(old_phrase) > 1:
            if self.all_gt[len(old_phrase)].actual_count(old_phrase) > 0:
                alpha, beta = self.get_alpha(old_phrase)
                pbo = alpha * self.get_pbo(wnew, old_phrase[1:])
            else:
                # If no data for (n-1)-gram, skip n-1 and use n-2
                pbo = self.get_pbo(wnew, old_phrase[1:])
        elif self.all_gt[len(old_phrase)].actual_count(old_phrase) > 0:
            alpha, beta = self.get_alpha(old_phrase)
            pbo = alpha * self.all_gt[len(old_phrase)].actual_count((wnew,)) / len(self.all_gt[len(old_phrase)].corpus)
        else:
            pbo = self.all_gt[len(old_phrase)].actual_count((wnew,)) / len(self.all_gt[len(old_phrase)].corpus)
            
        return pbo
