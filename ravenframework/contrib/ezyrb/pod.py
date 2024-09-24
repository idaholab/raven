# The MIT License (MIT)

# Copyright (c) 2016-2018 by EZyRB contributors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Adapted from https://github.com/mathLab/EZyRB

"""
Module for Proper Orthogonal Decomposition (POD).
Three different methods can be employed: Truncated Singular Value
Decomposition, Truncated Randomized Singular Value Decomposition, Truncated
Singular Value Decomposition via correlation matrix.
"""
import numpy as np

from .reduction import Reduction


class POD(Reduction):
    """
    Perform the Proper Orthogonal Decomposition.

    :param method: the implementation to use for the computation of the POD
        modes. Default is 'svd'.
    :type method: {'svd', 'randomized_svd', 'correlation_matrix'}
    :param rank: the rank for the truncation; If 0, the method computes
        the optimal rank and uses it for truncation; if positive interger,
        the method uses the argument for the truncation; if float between 0
        and 1, the rank is the number of the biggest singular values that
        are needed to reach the 'energy' specified by `svd_rank`; if -1,
        the method does not compute truncation. Default is 0. The `rank`
        parameter is available using all the available methods.
    :type rank: int or float
    :param int subspace_iteration: the number of subspace iteration in the
        randomized svd. It is available only using the 'randomized_svd'
        method. Default value is 1.
    :param int omega_rank: the number of columns of the Omega random
        matrix. If set to 0, the number of columns is equal to twice the
        `rank` (if it has explicitly passed as integer) or twice the number
        of input snapshots. Default value is 0. It is available only using
        the 'randomized_svd' method.
    :param bool save_memory: reduce the usage of the memory, despite an
        higher number of operations. It is available only using the
        'correlation_matrix' method. Default value is False.


    :Example:
        >>> pod = POD().fit(snapshots)
        >>> reduced_snapshots = pod.reduce(snapshots)
        >>> # Other possible constructors are ...
        >>> pod = POD('svd')
        >>> pod = POD('svd', rank=20)
        >>> pod = POD('randomized_svd', rank=-1)
        >>> pod = POD('randomized_svd', rank=0, subspace_iteration=3,
                        omega_rank=10)
        >>> pod = POD('correlation_matrix', rank=10, save_memory=False)
    """
    def __init__(self, method='svd', **kwargs):
        available_methods = {
            'svd': (self._svd, {
                'rank': -1
            }),
            'randomized_svd': (self._rsvd, {
                'rank': -1,
                'subspace_iteration': 1,
                'omega_rank': 0
            }),
            'correlation_matrix': (self._corrm, {
                'rank': -1,
                'save_memory': False
            }),
        }

        self._modes = None
        self._singular_values = None

        method = available_methods.get(method)
        if method is None:
            raise RuntimeError(
                f"Invalid method for POD. Please chose one among {', '.join(available_methods)}"
            )

        self.__method, args = method
        args.update(kwargs)

        for hyperparam, value in args.items():
            setattr(self, hyperparam, value)

    @property
    def modes(self):
        """
        The POD modes.

        :type: numpy.ndarray
        """
        return self._modes

    @property
    def singular_values(self):
        """
        The singular values

        :type: numpy.ndarray
        """
        return self._singular_values

    def fit(self, X):
        """
        Create the reduced space for the given snapshots `X` using the
        specified method

        :param numpy.ndarray X: the input snapshots matrix (stored by column)
        """
        self._modes, self._singular_values = self.__method(X)
        return self

    def transform(self, X):
        """
        Reduces the given snapshots.

        :param numpy.ndarray X: the input snapshots matrix (stored by column).
        """
        return self.modes.T.conj().dot(X)

    def inverse_transform(self, X):
        """
        Projects a reduced to full order solution.

        :type: numpy.ndarray
        """
        return self.modes.dot(X)

    def reduce(self, X):
        """
        Reduces the given snapshots.

        :param numpy.ndarray X: the input snapshots matrix (stored by column).

        .. note::

            Same as `transform`. Kept for backward compatibility.
        """
        return self.transform(X)

    def expand(self, X):
        """
        Projects a reduced to full order solution.

        :type: numpy.ndarray

        .. note::

            Same as `inverse_transform`. Kept for backward compatibility.
        """
        return self.inverse_transform(X)

    def _truncation(self, X, s):
        """
        Return the number of modes to select according to the `rank` parameter.
        See POD.__init__ for further info.

        :param numpy.ndarray X: the matrix to decompose.
        :param numpy.ndarray s: the singular values of X.

        :return: the number of modes
        :rtype: int
        """
        def omega(x):
            return 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43

        if self.rank == 0:
            beta = np.divide(*sorted(X.shape))
            tau = np.median(s) * omega(beta)
            rank = np.sum(s > tau)
        elif self.rank > 0 and self.rank < 1:
            cumulative_energy = np.cumsum(s**2 / (s**2).sum())
            rank = np.searchsorted(cumulative_energy, self.rank) + 1
        elif self.rank >= 1 and isinstance(self.rank, int):
            rank = self.rank
        else:
            rank = X.shape[1]

        return rank

    def _svd(self, X):
        """
        Truncated Singular Value Decomposition.

        :param numpy.ndarray X: the matrix to decompose.
        :return: the truncated left-singular vectors matrix, the truncated
            singular values array, the truncated right-singular vectors matrix.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray

        References:
        Gavish, Matan, and David L. Donoho, The optimal hard threshold for
        singular values is, IEEE Transactions on Information Theory 60.8
        (2014): 5040-5053.
        """
        U, s = np.linalg.svd(X, full_matrices=False)[:2]

        rank = self._truncation(X, s)
        return U[:, :rank], s[:rank]

    def _rsvd(self, X):
        """
        Truncated randomized Singular Value Decomposition.

        :param numpy.ndarray X: the matrix to decompose.
        :return: the truncated left-singular vectors matrix, the truncated
            singular values array, the truncated right-singular vectors matrix.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray

        References:
        Finding structure with randomness: probabilistic algorithms for
        constructing approximate matrix decompositions. N. Halko, P. G.
        Martinsson, J. A. Tropp.
        """
        if (self.omega_rank == 0 and isinstance(self.rank, int)
                and self.rank not in [0, -1]):
            omega_rank = self.rank * 2
        elif self.omega_rank == 0:
            omega_rank = X.shape[1] * 2
        else:
            omega_rank = self.omega_rank
        Omega = np.random.rand(X.shape[1], omega_rank)

        Y = np.dot(X, Omega)
        Q = np.linalg.qr(Y)[0]

        if self.subspace_iteration:
            for _ in range(self.subspace_iteration):
                Y_ = np.dot(X.T.conj(), Q)
                Q_ = np.linalg.qr(Y_)[0]
                Y = np.dot(X, Q_)
                Q = np.linalg.qr(Y)[0]

        B = np.dot(Q.T.conj(), X)
        Uy, s = np.linalg.svd(B, full_matrices=False)[:2]
        U = Q.dot(Uy)

        rank = self._truncation(X, s)
        return U[:, :rank], s[:rank]

    def _corrm(self, X):
        """
        Truncated POD calculated with correlation matrix.

        :param numpy.ndarray X: the matrix to decompose.
        :return: the truncated left-singular vectors matrix, the truncated
            singular values array, the truncated right-singular vectors matrix.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """

        if self.save_memory:
            corr = np.empty(shape=(X.shape[1], X.shape[1]))
            for i, i_snap in enumerate(X.T):
                for j, k_snap in enumerate(X.T):
                    corr[i, j] = np.inner(i_snap, k_snap)

        else:
            corr = X.T.dot(X)

        eigs, eigv = np.linalg.eigh(corr)

        ordered_idx = np.argsort(eigs)[::-1]
        eigs = eigs[ordered_idx]
        eigv = eigv[:, ordered_idx]
        s = np.sqrt(eigs[eigs > 0])
        rank = self._truncation(X, s)

        # compute modes
        eigv = eigv[:, eigs > 0]
        U = X.dot(eigv) / s

        return U[:, :rank], s[:rank]
