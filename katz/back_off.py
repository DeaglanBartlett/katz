import itertools

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

        # Create the required GoodTuring Objects
        self.all_gt = [None] * (max(all_len) + 1)
        for i in range(1, max(all_len) + 1):
            #            d = [s[:i] for s in corpus]
            #            # Deal with edge effect
            #            for j in range(all_len[0] - i):
            #                d = d + [corpus[-1][j+1:i+j+1]]
            #            self.all_gt[i] = GoodTuring(d)

            d = [
                s[-i:] for s in corpus if len(s) >= i
            ]  # The final n terms of the tuples
            d_start = [dd[:-1] for dd in d]  # The first n-1 terms of the n-tuple

            # These come in pairs to you can evaluate:
            #       C(w_{i-n+1) ... w_i)        [first item]
            #       C(w_{i-n+1) ... w_{i-1})    [second item]
            # Each pair in list is for a different n
            if i == 1:
                self.all_gt[i] = [GoodTuring(d), None]
            else:
                self.all_gt[i] = [GoodTuring(d), GoodTuring(d_start)]

        # Find all unique words in the corpus
        # self.words = list(sum(corpus, ()))
        self.words = list(itertools.chain(*corpus))
        self.words = sorted(set(self.words), key=self.words.index)

    def get_d(self, phrase):
        """
        Compute the amount of discounting found by Good-Turing estimation

        Args:
            :phrase (tuple): Collection of words to find discounting for

        Returns:
            :d (float): Good-Turing estimate for discounting

        """
        cstar = self.all_gt[len(phrase)][0].expected_count(
            phrase
        )  # C^*(w_{i-n+1} ... w_i)
        c = self.all_gt[len(phrase)][0].actual_count(phrase)  # C(w_{i-n+1} ... w_i)
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
        for w in self.words:
            if self.all_gt[len(phrase) + 1][0].actual_count(phrase + (w,)) == 0:
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
        seen, _ = self.sort_endings(old_phrase)

        beta = 0.0
        # Sum over w_i: C(w_{i-n+1} ... w_i) > 0 [i.e. seen]
        for w in seen:
            new_phrase = old_phrase + (w,)
            d = self.get_d(new_phrase)  # d_{w_{i-n+1} ... w_i}
            cnew = self.all_gt[len(new_phrase)][0].actual_count(
                new_phrase
            )  # C(w_{i-n+1} ... w_i)
            beta += d * cnew
        cold = self.all_gt[len(new_phrase)][1].actual_count(
            old_phrase
        )  # [1] since want to get  C(w_{i-n+1} ... w_{i-1})
        beta = 1 - beta / cold  # beta_{w_{i-n+1} ... w_{i-1}}

        # Expect len(seen) < len(unseen)
        # Economical to only run len(seen) times and then use normalisation of probs
        alpha = 1.0
        # Sum over w_i: C(w_{i-n+1} ... w_i) > 0 [i.e. seen]
        for w in seen:
            alpha -= self.get_pbo(w, old_phrase[1:])  # Pbo(w_i | w_{i-n+2} ... w_{i-1})
        alpha = beta / alpha  # alpha_{w_{i-n+1} ... w_{i-1}}

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
        cnew = self.all_gt[len(new_phrase)][0].actual_count(
            new_phrase
        )  # C(w_{i-n+1} ... w_i)

        if cnew > 0 and len(old_phrase) > 0:
            d = self.get_d(new_phrase)  # d_{w_{i-n+1} ... w_i}
            cold = self.all_gt[len(new_phrase)][1].actual_count(
                old_phrase
            )  # C(w_{i-n+1} ... w_{i-1})
            pbo = d * cnew / cold

        elif len(old_phrase) > 1:  # Never saw new phrase, and old phrase has length > 1
            if self.all_gt[len(new_phrase)][1].actual_count(old_phrase) > 0:
                # old_phrase appears as the start somewhere
                alpha, _ = self.get_alpha(old_phrase)  # alpha_{w_{i-n+1} ... w_{i-1}}
                pbo = alpha * self.get_pbo(
                    wnew, old_phrase[1:]
                )  # alpha_{w_{i-n+1} ... w_{i-1}} * Pbo(w_i | w_{i-n+2} ... w_{i-1})
            else:
                # If no data for (n-1)-gram, skip n-1 and use n-2
                pbo = self.get_pbo(
                    wnew, old_phrase[1:]
                )  # Pbo(w_i | w_{i-n+2} ... w_{i-1})

        elif (
            len(old_phrase) > 0
            and self.all_gt[len(new_phrase)][1].actual_count(old_phrase) > 0
        ):
            # Never saw new phrase, but have an old phrase of length 1 which appears somehwere in corpus
            alpha, _ = self.get_alpha(old_phrase)  # alpha_{w_{i-1}}
            pbo = (
                alpha
                * self.all_gt[1][0].actual_count((wnew,))
                / len(self.all_gt[1][0].corpus)
            )  # alpha_{w_{i-1}} * Pbo(w_{i} | . )

        else:
            pbo = self.all_gt[1][0].actual_count((wnew,)) / len(
                self.all_gt[1][0].corpus
            )

        return pbo


class NormalisedBackOff:
    """A fitted BackOff model with every conditional distribution made proper: negative
    or non-finite probabilities are set to zero and the rest divided by their sum over the
    vocabulary.

    The Katz model of Bartlett et al. (2023) is not always normalised. When the seen
    continuations of a context exhaust the support of the lower-order distribution, the mass
    reserved for unseen words is lost (the sum over words is then below one); and when the
    fitted Good-Turing slope is above -1, counts are inflated rather than discounted, which can
    give probabilities above one and negative back-off weights (in very small corpora); a
    zero denominator in a back-off weight gives NaN. KatzPrior applies this wrapper by
    default (normalise=True); BackOff itself is unchanged, so KatzPrior(..., smoothing='katz',
    root_context=False, normalise=False) gives the model of version 0.2.1, apart from the
    single-leaf correction.

    Args:
        :model (BackOff): The fitted Katz back-off model
        :vocab (list): Words over which each conditional distribution is normalised (default:
            the words of the training corpus)
        :follows (dict): Words that can only follow a given word, as {word: previous word}.
            Such a word has zero probability in any context whose word at follows_position
            is not that previous word (default: none)
        :follows_position (int, default=-1): Position in the context of the previous word

    Returns:
        NormalisedBackOff: Model with the same get_pbo interface
    """

    def __init__(self, model, vocab=None, follows=None, follows_position=-1):
        self.model = model
        self.words = list(model.words if vocab is None else vocab)
        self.follows = dict(follows or {})
        self.follows_position = follows_position
        self._dist = {}

    def __getattr__(self, name):
        # object.__getattribute__ so that a half-built instance (during unpickling or
        # copying, before `model` is set) raises AttributeError instead of recursing
        return getattr(object.__getattribute__(self, "model"), name)

    def get_pbo(self, wnew, old_phrase):
        """Normalised P(wnew | old_phrase); zero for words outside the vocabulary"""
        h = tuple(old_phrase)
        if h not in self._dist:
            p = {}
            for w in self.words:
                if w in self.follows and (
                    len(h) == 0 or h[self.follows_position] != self.follows[w]
                ):
                    p[w] = 0.0
                    continue
                v = float(self.model.get_pbo(w, h))
                p[w] = v if v > 0 and v != float("inf") else 0.0  # also NaN -> 0
            z = sum(p.values())
            self._dist[h] = {w: v / z for w, v in p.items()} if z > 0 else {}
        return self._dist[h].get(wnew, 0.0)


class FollowsRestriction:
    """A normalised language model (e.g. KneserNey) restricted to the words that may follow
    their context: a word in `follows` has zero probability unless the context word at
    `position` is the word it must follow, and the probabilities of the other words are
    divided by one minus the probability removed. The wrapped model's other attributes
    (e.g. set_discounts) are available unchanged.

    Args:
        :model: Model with a prob(word, context) method whose conditionals sum to one
        :follows (dict): {word: previous word}
        :position (int, default=-1): Position in the context of the previous word

    Returns:
        FollowsRestriction: Model with the same prob and get_pbo interface
    """

    def __init__(self, model, follows, position=-1):
        self.model = model
        self.follows = dict(follows)
        self.position = position

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "model"), name)

    def _excluded(self, h):
        return [
            w
            for w, prev in self.follows.items()
            if len(h) == 0 or h[self.position] != prev
        ]

    def prob(self, wnew, old_phrase=()):
        """Restricted P(wnew | old_phrase)"""
        h = tuple(old_phrase)
        excluded = self._excluded(h)
        if wnew in excluded:
            return 0.0
        p = self.model.prob(wnew, h)
        if excluded:
            p /= 1.0 - sum(self.model.prob(w, h) for w in excluded)
        return p

    def get_pbo(self, wnew, old_phrase):
        return self.prob(wnew, old_phrase)
