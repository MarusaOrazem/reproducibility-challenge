"""
Implementes graph edits as basis for graph edit networks.

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

import abc
import copy
import numpy as np

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class Edit(abc.ABC):

    @abc.abstractmethod
    def apply(self, A, X):
        """ Applies this edit to the given graph and returns a copy of the
        graph with the applied changes. The original graph remains unchanged.

        Parameters
        ----------
        A: class numpy.array
            An adjacency matrix.
        X: class numpy.array
            A node attribute list/matrix.

        Returns
        -------
        B: class numpy.array
            The output adjacency matrix.
        Y: class numpy.array
            The output attribute matrix.

        """
        pass

    @abc.abstractmethod
    def apply_in_place(self, A, X):
        """ Applies this edit to the given graph. Note that this changes the
        input arguments.

        Parameters
        ----------
        A: class numpy.array
            An adjacency matrix.
        X: class numpy.array
            A node attribute list/matrix.

        """
        pass

    @abc.abstractmethod
    def score(self, N):
        """ Transforms this edit to a N-dimensional score vector (for node operations)
        and a N x N score matrix (for edge operations) where an entry is +1 if the
        respective node spawns a new node/if the respective edge is inserted and
        -1 if the respective node/edge is deleted.

        Parameters
        ----------
        N: int
            The size of the graph.

        Returns
        -------
        delta: class numpy.array
            A N dimensional score vector with a +1 entry for new spawned nodes
            and a -1 entry for deleted nodes.
        Epsilon: class numpy.array
            a N x N score matrix with a +1 entry for new edges and a -1 entry
            for deleted edges.

        """
        pass

class NodeReplacement(Edit):

    def __init__(self, index, attribute):
        self._index = index
        self._attribute = attribute

    def apply(self, A, X):
        """ Replaces the label of node self._index with self._attribute.

        Parameters
        ----------
        A: class numpy.array
            An adjacency matrix.
        X: class numpy.array
            A node attribute list/matrix.

        Returns
        -------
        B: class numpy.array
            The output adjacency matrix.
        Y: class numpy.array
            The output attribute matrix.

        """
        A = np.copy(A)
        X = np.copy(X)
        self.apply_in_place(A, X)
        return A, X

    def apply_in_place(self, A, X):
        """ Deletes node self._index.

        Parameters
        ----------
        A: class numpy.array
            An adjacency matrix.
        X: class numpy.array
            A node attribute list/matrix.
        """
        X[self._index, :] = self._attribute

    def score(self, N):
        raise ValueError('unsupported')

    def __repr__(self):
        return 'rep(%d, %s)' % (self._index, self._attribute)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, NodeReplacement):
            return False
        if isinstance(self._attribute, np.ndarray):
            if not np.array_equal(self._attribute, other._attribute):
                return False
        else:
            if self._attribute != other._attribute:
                return False
        return self._index == other._index

class NodeDeletion(Edit):

    def __init__(self, index):
        self._index = index

    def apply(self, A, X):
        """ Deletes node self._index.

        Parameters
        ----------
        A: class numpy.array
            An adjacency matrix.
        X: class numpy.array
            A node attribute list/matrix.

        Returns
        -------
        B: class numpy.array
            The output adjacency matrix.
        Y: class numpy.array
            The output attribute matrix.

        """
        A = np.delete(np.delete(A, (self._index), axis=0), (self._index), axis=1)
        X = np.delete(X, (self._index), axis=0)
        return A, X

    def apply_in_place(self, A, X):
        """ Deletes node self._index.

        Parameters
        ----------
        A: class numpy.array
            An adjacency matrix.
        X: class numpy.array
            A node attribute list/matrix.

        """
        # This is unsupported for deletions
        raise ValueError('apply_in_place can not be supported for node deletions because the size of numpy matrices can not be changed in place')

    def score(self, N):
        """ Transforms this edit to a N-dimensional score vector, where
        entry Y[self._index] = -1 and every other entry is zero.

        Parameters
        ----------
        N: int
            The size of the graph.

        Returns
        -------
        delta: class numpy.array
            a N-dimensional score vector, where entry y[self._index] = -1 and
            every other entry is zero.
        Epsilon: class numpy.array
            a N x N zero matrix

        """
        y = np.zeros(N)
        y[self._index] = -1
        return y, np.zeros((N, N))

    def __repr__(self):
        return 'del(%d)' % (self._index)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, NodeDeletion) and self._index == other._index

class NodeInsertion(Edit):

    def __init__(self, index, attribute, directed = True):
        self._index = index
        self._attribute = attribute
        self._directed = directed

    def apply(self, A, X):
        """ Inserts a new node into the graph with self._attribute and
        connectes it to node self._index.

        Parameters
        ----------
        A: class numpy.array
            An adjacency matrix.
        X: class numpy.array
            A node attribute list/matrix.

        Returns
        -------
        B: class numpy.array
            The output adjacency matrix.
        Y: class numpy.array
            The output attribute matrix.

        """
        N, n = X.shape
        A_new = np.zeros((N + 1, N + 1))
        X_new = np.zeros((N + 1, n))

        A_new[:N, :][:, :N] = A
        A_new[self._index, N] = 1
        if(not self._directed):
            A_new[N, self._index] = 1

        X_new[:N, :] = X
        X_new[N, :]  = self._attribute
        return A_new, X_new

    def apply_in_place(self, A, X):
        """ Deletes node self._index.

        Parameters
        ----------
        A: class numpy.array
            An adjacency matrix.
        X: class numpy.array
            A node attribute list/matrix.
        """
        # This is unsupported for deletions
        raise ValueError('apply_in_place can not be supported for Node insertions because the size of numpy matrices can not be changed in place')

    def score(self, N):
        """ Transforms this edit to a N-dimensional score vector, where
        entry Y[self._index] = +1 and every other entry is zero.

        Parameters
        ----------
        N: int
            The size of the graph.

        Returns
        -------
        delta: class numpy.array
            a N-dimensional score vector, where entry y[self._index] = +1 and
            every other entry is zero.
        Epsilon: class numpy.array
            a N x N zero matrix

        """
        y = np.zeros(N)
        y[self._index] = +1
        return y, np.zeros((N, N))

    def __repr__(self):
        return 'ins(%d, %s)' % (self._index, self._attribute)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, NodeInsertion):
            return False
        if isinstance(self._attribute, np.ndarray):
            if not np.array_equal(self._attribute, other._attribute):
                return False
        else:
            if self._attribute != other._attribute:
                return False
        return self._index == other._index and self._directed == other._directed


class EdgeDeletion(Edit):

    def __init__(self, i, j, directed = True):
        self._i = i
        self._j = j
        self._directed = directed

    def apply(self, A, X):
        """ Deletes the edge from node self._i to node self._j.

        Parameters
        ----------
        A: class numpy.array
            An adjacency matrix.
        X: class numpy.array
            A node attribute list/matrix.

        Returns
        -------
        B: class numpy.array
            The output adjacency matrix.
        Y: class numpy.array
            The output attribute matrix.

        """
        A = np.copy(A)
        X = np.copy(X)
        self.apply_in_place(A, X)
        return A, X

    def apply_in_place(self, A, X):
        """ Deletes the edge from node self._i to node self._j.

        Parameters
        ----------
        A: class numpy.array
            An adjacency matrix.
        X: class numpy.array
            A node attribute list/matrix.

        """
        A[self._i, self._j] = 0
        if(not self._directed):
            A[self._j, self._i] = 0

    def score(self, N):
        """ Transforms this edit to a score matrix where the (i,j)th entry
        is set to -1 and everything else is zero.

        Parameters
        ----------
        N: int
            The size of the graph.

        Returns
        -------
        delta: class numpy.array
            A N dimensional zero vector.
        Epsilon: class numpy.array
            A N x N matrix where entry Y[i,j] = -1 and everything else is zero.

        """
        y = np.zeros(N)
        Y = np.zeros((N, N))
        Y[self._i, self._j] = -1
        if(not self._directed):
            Y[self._j, self._i] = -1
        return y, Y

    def __repr__(self):
        return 'del_edge(%d, %d)' % (self._i, self._j)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, EdgeDeletion) and self._i == other._i and self._j == other._j and self._directed == other._directed

class EdgeInsertion(Edit):

    def __init__(self, i, j, directed = True):
        self._i = i
        self._j = j
        self._directed = directed

    def apply(self, A, X):
        """ Inserts a new edge from node self._i to node self._j.

        Parameters
        ----------
        A: class numpy.array
            An adjacency matrix.
        X: class numpy.array
            A node attribute list/matrix.

        Returns
        -------
        B: class numpy.array
            The output adjacency matrix.
        Y: class numpy.array
            The output attribute matrix.

        """
        A = np.copy(A)
        X = np.copy(X)
        self.apply_in_place(A, X)
        return A, X

    def apply_in_place(self, A, X):
        """ Inserts a new edge from node self._i to node self._j.

        Parameters
        ----------
        A: class numpy.array
            An adjacency matrix.
        X: class numpy.array
            A node attribute list/matrix.

        """
        A[self._i, self._j] = 1
        if(not self._directed):
            A[self._j, self._i] = 1

    def score(self, N):
        """ Transforms this edit to a score matrix where the (i,j)th entry
        is set to +1 and everything else is zero.

        Parameters
        ----------
        N: int
            The size of the graph.

        Returns
        -------
        delta: class numpy.array
            A N dimensional zero vector.
        Epsilon: class numpy.array
            A N x N matrix where entry Y[i,j] = +1 and everything else is zero.

        """
        y = np.zeros(N)
        Y = np.zeros((N, N))
        Y[self._i, self._j] = +1
        if(not self._directed):
            Y[self._j, self._i] = +1
        return y, Y

    def __repr__(self):
        return 'ins_edge(%d, %d)' % (self._i, self._j)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, EdgeInsertion) and self._i == other._i and self._j == other._j and self._directed == other._directed

def apply_script(script, A, X):
    """ Applies the given list of edits to the given graph.

    Parameters
    ----------
    script: list of class.Edit
        A list of graph edits.
    A: ndarray of size M x M
        The adjacency matrix of the input graph
    X: ndarray of size M x n
        The attribute matrix of the input graph

    Returns
    -------
    B: ndarray of size N x N
        The adjacency matrix of the output graph
    Y: ndarray of size N x n
        The attribute matrix of the output graph

    """
    if len(script) == 0:
        return A, X

    # apply the first edit as copy
    B, Y = script[0].apply(A, X)

    # apply all remaining edits in place (if possible)
    for edit in script[1:]:
        try:
            edit.apply_in_place(B, Y)
        except ValueError as ex:
            B, Y = edit.apply(B, Y)

    return B, Y


def mapping_to_edits(A, X, B, Y, psi, directed = True, return_full_psi = False):
    """ Computes an edit script that transforms graph (X, A) into graph
    (Y, B) based on the graph mapping psi.

    Parameters
    ----------
    A: ndarray of size M x M
        The adjacency matrix of the first graph.
    X: ndarray of size M x n
        The attribute matrix of the first graph.
    B: ndarray of size N x N
        The adjacency matrix of the second graph.
    Y: ndarray of size N x n
        The attribute matrix of the second graph.
    psi: ndarray of size M
        A mapping from nodes in the first to nodes in the second graph.
        Deleted nodes are marked with a negative number or a number >= N.
    directed: bool (default = True)
        If set to False, all edits are undirected.
    return_full_psi: bool (default = False)
        If set to True, a completed version of psi is returned which also
        contains inserted nodes.

    Returns
    -------
    script: list
        A list of graph edits that transforms the first into the second graph.

    """
    M, n = X.shape
    if A.shape[0] != M or A.shape[1] != M:
        raise ValueError('Expected a %d x %d adjacency matrix for the first graph but got a %d x %d matrix.' % (M, M, A.shape[0], A.shape[1]))

    if Y.shape[1] != n:
        raise ValueError('Expected %d columns in the attribute matrix of the second graph but got %d columns.' % (n, Y.shape[1]))

    N, _ = Y.shape
    if B.shape[0] != N or B.shape[1] != N:
        raise ValueError('Expected a %d x %d adjacency matrix for the second graph but got a %d x %d matrix.' % (N, N, B.shape[0], B.shape[1]))

    # first, normalize the format for deleted node to be N
    psi = np.copy(psi)
    psi[psi < 0] = N

    # construct the inverse map
    psi_inv = np.full(N, M)
    for i in range(M):
        if psi[i] >= N:
            continue
        psi_inv[psi[i]] = i

    # now, start constructing the script
    script = []
    # then, start with replacements
    for i in range(M):
        if psi[i] >= N:
            continue
        if np.mean(np.abs(X[i, :] - Y[psi[i], :])) > 1E-3:
            script.append(NodeReplacement(i, Y[psi[i], :]))

    # next, apply node insertions. For this purpose, we extend psi to the
    # inserted nodes as well
    psi_ins = []
    A_ins   = []
    for j in range(N):
        if psi_inv[j] < M:
            continue
        # Identify a parent, ideally a parent that is already connected
        # to the new node in the target graph.
        present = np.concatenate((B[:j, j], B[psi_inv < M, j]), 0)
        present_idxs = np.concatenate((np.arange(j), np.where(psi_inv < M)[0]), 0)
        if np.any(present):
            p = present_idxs[np.where(present)[0][0]]
            i = psi_inv[p]
        else:
            # of no such parent exists, just use the first node
            i = 0

        if i >= M + len(A_ins):
            print('psi: %s' % str(psi))
            print('psi_inv: %s' % str(psi_inv))
            print('present: %s' % str(present))
            print('present_idxs: %s' % str(present_idxs))
            raise ValueError('Internal error: Tried to perform an insertion from a not-yet present node.')

        # append the node insertion
        script.append(NodeInsertion(i, Y[j, :], directed = directed))
        # append a new entry for the mapping
        psi_ins.append(j)
        # set the according entry in the inverse mapping
        psi_inv[j] = M + len(psi_ins) - 1
        # add a row to the adjacency matrix
        ains = np.zeros(M+N)
        ains[i] = 1.
        A_ins.append(ains)

    # complete the mapping
    psi_ins = np.array(psi_ins)
    K = len(psi_ins)
    psi = np.concatenate((psi, psi_ins), 0)
    if K > 0:
        # complete the adjacency matrix
        A_ins = np.stack(A_ins, 0)[:, :M+K]
        Afull = np.zeros((M + K, M + K))
        Afull[:M, :M] = A
        Afull[:M, M:] = A_ins[:, :M].T
        if not directed:
            Afull[M:, :]  = A_ins
    else:
        Afull = A

    psi = psi.astype(int)

    # next, perform edge insertions
    for i in range(M+K):
        if psi[i] >= N:
            continue
        lo = 0 if directed else i+1
        for j in range(lo, M+K):
            if psi[j] >= N:
                continue
            if Afull[i, j] < 0.5 and B[psi[i], psi[j]] > 0.5:
                script.append(EdgeInsertion(i, j, directed = directed))
    # next, perform edge deletions
    for i in range(M+K):
        if psi[i] >= N:
            continue
        lo = 0 if directed else i+1
        for j in range(lo, M+K):
            if psi[j] >= N:
                continue
            if Afull[i, j] > 0.5 and B[psi[i], psi[j]] < 0.5:
                script.append(EdgeDeletion(i, j, directed = directed))

    # finally, perform node deletions
    for i in range(M-1, -1, -1):
        if psi[i] < N:
            continue
        script.append(NodeDeletion(i))

    psi = psi[psi < N]

    # return the script
    if return_full_psi:
        return script, psi
    else:
        return script
