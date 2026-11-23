# DISCLAIMER:
# This file includes code generated with the assistance of an AI system.
# The code has been reviewed, tested, and verified by a developer;
# however, it is provided without warranty. Users are responsible for
# ensuring correctness, security, and compliance in their environment.


import numpy as np
from scipy.stats import beta


class PertDist:
    """
    PERT distribution implemented using scipy.stats.beta.

    Parameters
    ----------
    a : float
        Minimum value
    b : float
        Maximum value
    m : float
        Most likely (mode)
    lamb : float, optional
        Shape parameter (default = 4)
    """

    def __init__(self, a, b, m, lamb=4.0):
        if not (a <= m <= b):
            raise ValueError("Require a <= m <= b")
        if b <= a:
            raise ValueError("Require b > a")

        self.a = float(a)
        self.b = float(b)
        self.m = float(m)
        self.lamb = float(lamb)

        self.alpha = 1.0 + lamb * (m - a) / (b - a)
        self.beta = 1.0 + lamb * (b - m) / (b - a)

        self._beta_dist = beta(self.alpha, self.beta)
        self._mean = (self.a + self.b + self.lamb * self.m) / (self.lamb + 2)
        self._var = ((self._mean-self.a) * (self.b-self._mean)) / (self.lamb+3)

    # ---- Core distribution methods ----

    def pdf(self, x):
        x = np.asarray(x)
        y = (x - self.a) / (self.b - self.a)
        return self._beta_dist.pdf(y) / (self.b - self.a)

    def cdf(self, x):
        x = np.asarray(x)
        y = (x - self.a) / (self.b - self.a)
        return self._beta_dist.cdf(y)

    def ppf(self, q):
        y = self._beta_dist.ppf(q)
        return self.a + y * (self.b - self.a)

    def rvs(self, size=None, random_state=None):
        y = self._beta_dist.rvs(size=size, random_state=random_state)
        return self.a + y * (self.b - self.a)

    # ---- Moments ----

    def mean(self):
        return self._mean

    def var(self):
        return self._var

    def std(self):
        return np.sqrt(self._var)

    def stats(self):
        return self.mean(), self.var()

    def mode(self):
        return self.m

    def __repr__(self):
        return (f"PERT(a={self.a}, b={self.b}, m={self.m}, "
                f"lamb={self.lamb})")

