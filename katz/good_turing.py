import collections
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

class GoodTuring:

    def __init__(self, corpus):
    
        # Store the data
        self.corpus = corpus
    
        # Dict for frequency of each element
        self.R = dict(collections.Counter(corpus))
        
        # Array for frequency of frequencies
        Nr = dict(collections.Counter(list(self.R.values())))
        Nr = np.array([list(Nr.keys()), list(Nr.values())])
        idx = np.argsort(Nr[0,:])
        self.Nr = Nr[:,idx]
        
        # Apply smoothing to get Zr
        Zr = np.zeros(self.Nr.shape[1], dtype=float)
        q = np.concatenate(([0], self.Nr[0,:-2]))
        t = self.Nr[0,1:]
        Zr[:-1] = self.Nr[1,:-1] / (0.5 * (t - q))
        Zr[-1] = self.Nr[1,-1] / (self.Nr[0,-1] - self.Nr[0,-2])
        self.Zr = Zr
        
        # Apply linear regression
        res = scipy.stats.linregress(np.log(self.Nr[0,:]), np.log(self.Zr))
        self.slope = res.slope
        self.intercept = res.intercept
        
    def get_S(self, r):
        return np.exp(self.slope * np.log(r) + self.intercept)
        
    def actual_count(self, word):
        if word in self.R.keys():
            return self.R[word]
        else:
            return 0

    def expected_count(self, word):
        k = self.actual_count(word)
        return (k+1) * self.get_S(k+1) / self.get_S(k)

    def plot_fit(self):
        plt.figure()
        plt.loglog(self.Nr[0,:], self.Zr, '.')
        x = np.logspace(np.log10(self.Nr[0,0]), np.log10(self.Nr[0,-1]))
        y = self.get_S(x)
        plt.loglog(x, y)
        plt.xlabel(r'$r$')
        plt.ylabel(r'$Z_r$')
        plt.tight_layout()
        plt.show()
        plt.clf()
