import collections
import numbers

import numpy as np


def chen_goodman_discounts(count_of_counts, fallback=0.5, min_discount=1e-3):
    """
    Modified Kneser-Ney discounts (D1, D2, D3+) estimated from count-of-counts
    following Chen & Goodman (1999):

        Y = n1 / (n1 + 2 n2),   D_r = r - (r + 1) Y n_{r+1} / n_r   (r = 1, 2, 3)

    Each D_r is clipped to [min_discount, r] so that every discounted count stays
    non-negative and some probability mass is always reserved for lower orders.
    If a discount is undefined (a vanishing count-of-counts), it falls back to the
    single absolute discount Y, or to `fallback` if Y is itself undefined.

    Args:
        :count_of_counts (dict): Maps a count r to the number of n-grams seen exactly r times
        :fallback (float): Discount used when no estimate is possible
        :min_discount (float): Lower bound applied to every discount

    Returns:
        :D (np.ndarray): Array of the three discounts (D1, D2, D3+)
    """
    n = [count_of_counts.get(r, 0) for r in range(5)]
    Y = n[1] / (n[1] + 2 * n[2]) if (n[1] > 0 and n[2] > 0) else None
    D = np.empty(3)
    for r in (1, 2, 3):
        if Y is not None and n[r] > 0:
            d = r - (r + 1) * Y * n[r + 1] / n[r]
        else:
            d = Y if Y is not None else fallback
        D[r - 1] = np.clip(d, min_discount, r)
    return D


class KneserNey:

    def __init__(
        self,
        corpus,
        order=None,
        vocab=None,
        discounts="mkn",
        discount_scale=1.0,
        cache_size=2**20,
    ):
        """Interpolated Kneser-Ney model for predicting the final element of a tuple
        given the elements which precede it. This is a drop-in alternative to
        katz.back_off.BackOff (same get_pbo interface). With one discount per order this
        is interpolated Kneser-Ney with absolute discounting; with count-dependent
        discounts (D1, D2, D3+, e.g. the Chen-Goodman estimates 'mkn') it is modified
        Kneser-Ney.

        Each tuple in the corpus is one prediction event: its last element is the
        predicted word and the preceding elements are its context (most distant first).
        Tuples may have different lengths; a tuple shorter than `order` is one whose
        context cannot be extended further (e.g. near the root of an expression tree),
        so its n-gram is counted with its raw count, exactly like n-grams starting with
        a sentence-boundary symbol in standard Kneser-Ney (Chen & Goodman 1999; KenLM).
        At every lower order the counts are continuation counts N_{1+}(. g), i.e. the
        number of distinct words seen immediately before the n-gram g. The recursion
        terminates in a uniform distribution over the vocabulary, so every word in
        the vocabulary receives a non-zero probability in every context. An empty corpus
        (which requires `order` and `vocab`) gives the uniform distribution.

        Args:
            :corpus (list of tuples): The prediction events (tuples of hashable words)
            :order (int): Maximum n-gram order (context length + 1). Defaults to the longest tuple. Longer tuples are truncated to their final `order` elements.
            :vocab (iterable): Words which may be predicted, in addition to those predicted in the corpus. Probabilities are normalised over this set.
            :discounts: 'mkn' (three Chen-Goodman discounts per order), 'kn' (one absolute discount Y per order), a float (used for every order and count), an array of shape (order,) (one discount per order) or of shape (order, 3) (D1, D2, D3+ per order)
            :discount_scale (float): Positive factor applied to the discounts (after estimation, before clipping to their maximum D_r = r)
            :cache_size (int): Maximum number of cached probabilities (the cache is cleared when it is full); 0 disables caching

        Returns:
            KneserNey: The smoothed language model
        """

        corpus = [tuple(s) for s in corpus if len(s) > 0]
        if order is None:
            if len(corpus) == 0:
                raise ValueError("An empty corpus requires the order to be given")
            order = max(len(s) for s in corpus)
        if int(order) != order or order < 1:
            raise ValueError(f"order must be a positive integer, not {order}")
        self.order = int(order)
        if (
            isinstance(cache_size, bool)
            or not isinstance(cache_size, numbers.Integral)
            or cache_size < 0
        ):
            raise ValueError(
                f"cache_size must be a non-negative integer, not {cache_size!r}"
            )
        self.cache_size = int(cache_size)
        events = [s[-self.order :] for s in corpus]

        # Adjusted counts: raw count at the highest order available for each event,
        # continuation (distinct left-neighbour) counts at all lower orders
        raw = [collections.Counter() for _ in range(self.order + 1)]
        left = [collections.defaultdict(set) for _ in range(self.order + 1)]
        for s in events:
            raw[len(s)][s] += 1
            for k in range(1, len(s)):
                left[k][s[-k:]].add(s[-k - 1])

        self.counts = [None] * (self.order + 1)
        self.total = [None] * (self.order + 1)  # sum_w c(h w)
        self.nr = [None] * (self.order + 1)  # number of w with c(h w) = 1, 2, >=3
        self.count_of_counts = [None] * (self.order + 1)
        for k in range(1, self.order + 1):
            c = collections.Counter(raw[k])
            for g, s in left[k].items():
                c[g] += len(s)
            self.counts[k] = dict(c)
            tot = collections.defaultdict(int)
            nr = collections.defaultdict(lambda: np.zeros(3))
            for g, a in c.items():
                tot[g[:-1]] += a
                nr[g[:-1]][min(a, 3) - 1] += 1
            self.total[k] = dict(tot)
            self.nr[k] = dict(nr)
            self.count_of_counts[k] = collections.Counter(c.values())

        # Vocabulary: every word predicted anywhere appears at order 1
        observed = {g[0] for g in self.counts[1]}
        extra = set() if vocab is None else set(vocab)
        self.vocab = sorted(observed | extra, key=str)
        if len(self.vocab) == 0:
            raise ValueError("Empty vocabulary: give a corpus or a vocab")
        self.vocab_set = set(self.vocab)
        self.words = self.vocab  # for compatibility with BackOff

        self.set_discounts(discounts, discount_scale)

    def set_discounts(self, discounts="mkn", discount_scale=1.0):
        """
        Set the discounts without recounting the corpus (e.g. when tuning them)

        Args:
            :discounts: See the class constructor
            :discount_scale (float): See the class constructor

        Returns:
            :None
        """
        self.discounts = self._make_discounts(discounts, discount_scale)
        self._cache = {}

    def _make_discounts(self, discounts, scale):
        scale = float(scale)
        if not (np.isfinite(scale) and scale > 0):
            raise ValueError(f"discount_scale must be finite and positive, not {scale}")
        if not isinstance(discounts, str):
            d = np.asarray(discounts, dtype=float)
            allowed = [(), (self.order,), (self.order, 3)]
            if d.shape not in allowed:
                raise ValueError(
                    f"discounts must be a scalar or have shape ({self.order},) or "
                    f"({self.order}, 3), not {d.shape}"
                )
            if not np.all(np.isfinite(d) & (d > 0)):
                raise ValueError(
                    "discounts must be finite and positive (a zero discount gives "
                    "unseen words zero probability)"
                )
        D = np.zeros((self.order + 1, 3))
        for k in range(1, self.order + 1):
            if isinstance(discounts, str):
                if discounts == "mkn":
                    D[k] = chen_goodman_discounts(self.count_of_counts[k])
                elif discounts == "kn":
                    n1 = self.count_of_counts[k].get(1, 0)
                    n2 = self.count_of_counts[k].get(2, 0)
                    D[k] = n1 / (n1 + 2 * n2) if (n1 > 0 and n2 > 0) else 0.5
                else:
                    raise ValueError(f"Unknown discount method: {discounts}")
            else:
                d = np.asarray(discounts, dtype=float)
                if d.ndim == 0:
                    D[k] = d
                elif d.ndim == 1:
                    D[k] = d[k - 1]
                else:
                    D[k] = d[k - 1]
        D[1:] *= scale
        D[1:] = np.minimum(D[1:], np.array([1.0, 2.0, 3.0]))  # D_r <= r
        return D

    def prob(self, wnew, old_phrase=()):
        """
        Compute the probability of word wnew given the preceding words old_phrase

        Args:
            :wnew: The word whose probability we want
            :old_phrase (tuple): The preceding words (most distant first). Only the final order-1 are used.

        Returns:
            :p (float): The conditional probability P(wnew | old_phrase). Zero if wnew is not in the vocabulary.
        """
        h = tuple(old_phrase)
        h = h[-(self.order - 1) :] if self.order > 1 else ()  # the last order-1 words
        key = (wnew, h)
        if key in self._cache:
            return self._cache[key]
        if wnew not in self.vocab_set:
            return 0.0

        p = 1.0 / len(self.vocab)
        for k in range(1, len(h) + 2):
            ctx = h[len(h) - (k - 1) :] if k > 1 else ()
            T = self.total[k].get(ctx, 0)
            if T == 0:
                continue  # context never seen at this order: keep lower-order estimate
            a = self.counts[k].get(ctx + (wnew,), 0)
            D = self.discounts[k]
            gamma = np.dot(D, self.nr[k][ctx]) / T
            d = D[min(a, 3) - 1] if a > 0 else 0.0
            p = (a - d) / T + gamma * p
        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                self._cache.clear()
            self._cache[key] = p
        return p

    def get_pbo(self, wnew, old_phrase):
        """
        Compute the probability for word wnew given the preceding set of words old_phrase.
        Alias of prob, so that this class can replace katz.back_off.BackOff.

        Args:
            :wnew: The new word we wish to know the probability of obtaining
            :old_phrase (tuple): The preceding phrase

        Returns:
            :pbo (float): The conditional probability P(wnew|old_phrase)
        """
        return self.prob(wnew, old_phrase)
