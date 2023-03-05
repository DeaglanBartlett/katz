from .good_turing import GoodTuring


class BackOff:

    def __init__(self, corpus):
    
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
        cstar = self.all_gt[len(phrase)].expected_count(phrase)
        c = self.all_gt[len(phrase)].actual_count(phrase)
        return cstar / c
        
    def sort_endings(self, phrase):
        seen = []
        unseen = []
        for i, w in enumerate(self.words):
            if self.all_gt[len(phrase)+1].actual_count(phrase + (w,)) == 0:
                unseen.append(w)
            else:
                seen.append(w)
                
        return seen, unseen
        
    def get_alpha(self, old_phrase):
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
