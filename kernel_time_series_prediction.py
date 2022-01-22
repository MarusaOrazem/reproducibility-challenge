"""
Implements time seried prediction in the kernel space according to the paper

Paaßen, B., Göpfert, C., & Hammer, B. (2018).
Time Series Prediction for Graphs in Kernel and Dissimilarity Spaces.
Neural Processing Letters. doi:doi:10.1007/s11063-017-9684-5
arXiv:https://arxiv.org/abs/1704.06498

To perform the time series prediction, initialize a KernelTreePredictor,
call its 'fit' method on a training set of tree time series and then call
predict on new time series.

The predictor does not only predict in kernel space but also performs a mapping
back to the space of trees via the scheme proposed in

Paaßen, B., Hammer, B., Price, T., Barnes, T., Gross,S., & Pinkwart, N. (2018).
The Continuous Hint Factory - Providing Hints in Vast and Sparsely Populated
Edit Distance Spaces. Journal of Educational Datamining.
URL: https://jedm.educationaldatamining.org/index.php/JEDM/article/view/158

"""

# Copyright (C) 2019-2021
# Benjamin Paaßen
# The University of Sydney

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from sklearn.base import BaseEstimator
from edist.multiprocess import pairwise_distances_symmetric
import edist.ted as ted
import edist.tree_edits as te
import edist.tree_utils as tu
import graph_edits as ge

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class KernelTreePredictor(BaseEstimator):
    """ A time series predictor for time series of trees based on the edit
    distance and a radial basis function kernel and maps the prediction from
    kernel space back to the space of trees based on an edit distance trick.

    In particular, let x_1^k, ..., x_{T_j}^k be the kth time series in the
    training data. First, we re-order these time series to tuples
    (x_t^j, x_{t+1}^j) = (x_i, y_i).
    Let \{(x_i, y_j)\} now be the overall set of these tuples.

    Second, we compute all pairwise tree edit distances d(x_i, x_j).

    Third, we transform these into a similarity matrix K with entries

    .. math:: K_{i, j} = exp[-d(x_i, x_j)/(2 * \\psi ** 2)]

    where self.psi is the bandwidth parameter of the radial basis function
    kernel.

    Note that this matrix may be indefinite because the tree edit distance
    is generally not Euclidean. In other words, K may not be a kernel.
    However, the remaining scheme will still work even with slightly negative
    eigenvalues.

    Finally, we add the noise variance to the diagonal and invert the matrix.

    .. math:: \\tilde K = (K + \\sigma^2 \cdot I)^{-1}

    We store K tilde as self._Kinv.

    When we wish to predict, we first compute the edit distances of current
    tree x to all training trees d(x, x_i).

    Second, we transform these into a kernel vector k with entries

    .. math:: k_i = exp[-d(x, x_i)/(2 * self.sigma ** 2)]

    Third, we apply the predictive formula

    .. math:: \\alpha = \\tilde K \cdot \vec k

    The coefficients alpha define a linear combination of change vectors in
    the kernel space. In particular, we can write our prediction in the kernel
    space as

    .. math:: \\vec y = \\phi(x) + \\sum_i \\alpha_i \\cdot [\\phi(y_i) - \\phi(x_i)]

    where phi is the mapping into the kernel space.

    If we now wish to infer the tree in primal space, this problem can be
    re-written as follows.

    .. math:: \\min_y \\quad ||\\vec y - \\phi(y)||^2

    In other words, we are looking for the tree that is as close as possible to
    our predicted vector in kernel space. Assuming that the Euclidean distance
    in kernel space is approximatively equivalent to the tree edit distance, we
    can re-write this problem as

    .. math:: \\min_y \\quad d(x, y)^2 + \\sum_i \\alpha_i [d(y_i, y)^2 - d(y_i, y)^2]

    where d is the tree edit distance. We can apply a greedy scheme for this
    optimization problem by applying edits toward y_i where alpha_i is largest
    until the loss does not decrease anymore, then applying edits toward y_i
    where alpha_i is second-largest until the loss does not decrease anymore,
    and so on until there are no coefficients left.

    Attributes
    ----------
    psi: float (> 0, optional, default=None)
        The radial basis function kernel bandwidth. Defaults to half the
        average edit distance in the data set.
    sigma: float (>= 0, optional, default=1E-3)
        The noise standard deviation / the regularization parameter for kernel
        regression.
    _X: list
        A list of input training trees.
    _Y: list
        A list of output training trees.
    _time_series: list
        A list of time series of trees, i.e. the training data.
    _Kinv: array_like
        The inverted kernel matrix for the training data.

    """
    def __init__(self, psi = None, sigma = 1E-3):
        self.psi = psi
        self.sigma = sigma

    def fit(self, time_series):
        """ Fits this predictor to the given time series of trees.

        In more detail, this method computes the pairwise edit distances
        and inverse kernel matrix for all trees in the given time series.

        Parameters
        ----------
        time_series: list
            A list of time series, where each time series is in turn a list
            of trees, each given in node list/adjacency list format.

        Returns
        -------
        class kernel_time_series_prediction.KernelTreePredictor
            self

        """
        # prepare _X and _Y
        self._X = []
        self._Y = []
        for seq in time_series:
            if len(seq) == 0:
                continue
            for t in range(len(seq)-1):
                self._X.append((seq[t][0], seq[t][1]))
                self._Y.append((seq[t+1][0], seq[t+1][1]))
            self._X.append((seq[-1][0], seq[-1][1]))
            self._Y.append((seq[-1][0], seq[-1][1]))
        # store the original data as well
        self._time_series = time_series
        # compute all pairwise edit distance values
        D = pairwise_distances_symmetric(self._X, ted.ted)
        # adjust psi if it is not set
        if self.psi is None:
            self.psi = np.mean(D) * 0.5
        # transform to similarity matrix
        K = np.exp(-0.5 * D ** 2 / self.psi ** 2)
        # invert
        self._Kinv = np.linalg.inv(K + self.sigma ** 2 * np.eye(len(K)))

    def predict(self, nodes, adj):
        """ Predicts the next time step for a given input tree, both in
        kernel space and in tree space.

        Parameters
        ----------
        nodes: list
            A list of nodes for the input tree.
        adj:
            A list of edges for the input tree.

        Returns
        -------
        alpha: array_like
            A len(self._X) dimensional array containing the linear coefficients
            that represent the predicted position in kernel space.
        nodes: list
            A list of nodes for the predicted tree.
        adj:
            A list of edges for the predicted tree.

        """
        # compute the edit distances to all training data
        d = np.zeros(len(self._X))
        for i in range(len(self._X)):
            d[i] = ted.ted((nodes, adj), self._X[i])
        # compute the kernel vector
        k = np.exp(-0.5 * d ** 2 / self.psi ** 2)
        # compute the linear coefficients
        alpha = np.dot(self._Kinv, k)

        # approximate the tree in primal space via a greedy edit distance
        # scheme. In particular, we perform edits toward all training trees
        # (trees with largest coefficients alpha first) until the distance
        # to the prediction in kernel space does not decrease anymore.

        # sort the coefficients
        idxs = np.flip(np.argsort(np.abs(alpha)))
        # compute the initial squared distance in kernel space
        d2_x = d ** 2
        d2_y = np.zeros(len(self._Y))
        i = 0
        for seq in self._time_series:
            for t in range(len(seq)-1):
                d2_y[i] = d2_x[i+1]
                i += 1
            d2_y[i] = d2_x[i]
            i += 1
        loss = np.dot(d2_y, alpha) - np.dot(d2_x, alpha)

        # iterate over all coefficients, starting with the largest
        out_nodes = nodes
        out_adj   = adj
        for i in idxs:
            # retrieve the edits between the input tree and the training tree
            if alpha[i] < 0.:
                training_tree = self._X[i]
            else:
                training_tree = self._Y[i]
            alignment = ted.ted_backtrace((out_nodes, out_adj), training_tree)
            script = te.alignment_to_script(alignment, out_nodes, out_adj, training_tree[0], training_tree[1])
            # apply the script until the loss does not decrease anymore
            for edit in script:
                next_nodes, next_adj = edit.apply(out_nodes, out_adj)
                # compute the updated tree edit distances
                next_d2_x = np.zeros(len(self._X))
                next_d2_y = np.zeros(len(self._Y))
                i = 0
                for seq in self._time_series:
                    next_d2_x[i] = ted.ted((next_nodes, next_adj), (seq[0][0], seq[0][1])) ** 2
                    i += 1
                    for t in range(1, len(seq)):
                        next_d2_x[i] = ted.ted((next_nodes, next_adj), (seq[t][0], seq[t][1])) ** 2
                        next_d2_y[i-1] = next_d2_x[i]
                        i += 1
                    next_d2_y[i-1] = next_d2_x[i-1]
                next_loss = ted.ted((nodes, adj), (next_nodes, next_adj)) ** 2 + np.dot(next_d2_y, alpha) - np.dot(next_d2_x, alpha)
                if next_loss < loss:
                    loss = next_loss
                    out_nodes = next_nodes
                    out_adj  = next_adj
                elif not isinstance(edit, te.Replacement):
                    # if the edit would change the index structure, we break
                    # off the search for the current script and rather check
                    # the next one
                    break
        return alpha, out_nodes, out_adj
