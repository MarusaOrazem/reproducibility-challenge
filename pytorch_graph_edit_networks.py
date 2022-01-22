"""
Implements graph convolutional neural network layers that assign latent
representations to both nodes _and_ edges and a loss function for edit
scores.

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

import torch
import graph_edits as ge

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class GCN(torch.nn.Module):
    """ Implements a simple graph convolutional network variation that computes
    a latent representation for nodes in a graph as follows:

    x = W * x + U * sum{incoming neighbors y} + V * sum{outgoing neighbors z} + b

    where U, V, and W are matrices of size self._dim_out x self._dim_in and
    where b is a self._dim_out dimensional bias vector.

    Attributes
    ----------
    _dim_in: int
        The input dimensionality for this layer.
    _dim_out: int
        The output dimensionality for this layer.
    _U: class torch.nn.Linear
        implements the U matrix.
    _V: class torch.nn.Linear
        implements the V matrix
    _W: class torch.nn.Linear
        implements the W matrix.

    """
    def __init__(self, dim_in, dim_out):
        super(GCN, self).__init__()
        self._dim_in  = dim_in
        self._dim_out = dim_out
        self._U = torch.nn.Linear(self._dim_in, self._dim_out)
        self._V = torch.nn.Linear(self._dim_in, self._dim_out)
        self._W = torch.nn.Linear(self._dim_in, self._dim_out)

    def forward(self, X, A):
        """ Computes the next layer of node representations for the graph
        with current representation matrix X and adjacency matrix A.

        Parameters
        ----------
        X: class torch.Tensor
            the len(A) x self._dim_in matrix of current node representations.
        A: class torch.Tensor
            the len(A) x len(A) adjacency matrix.

        Returns
        -------
        X: class Torch.Tensor
            the new len(A) x self._dim_out matrix of node features

        """
        return self._W(X) + self._U(torch.mm(A.t(), X)) + self._V(torch.mm(A, X))

class GEN(torch.nn.Module):
    """ Computes a node edit score and an edge edit score for each node
    and each potential edge in an input graph. The node edit score is
    computes via graph convolutional neural network layers. The edge edit
    score is computed from the hidden node representations via the formula

    e_ij = u * x_i + v * x_j + w * (x_i * x_j)

    where u and v are vectors of size _dim_hid[-1] and w is a single number

    Attributes
    ----------
    _num_layers: int
        The number of GCN layers.
    _dim_in: int
        The input dimensionality of node representations.
    _dim_hid: int
       The number of hidden neurons in each layer to compute node
        representations. Can be a list of length _num_layers-1 or a single
        number.
    _nonlin: class torch.nn.Module (default = torch.nn.ReLU())
        The nonlinearity applied after each layer.
    _filter_edge_edits: boolean or int (default = False)
        If set to True, this class sets up two binary classifiers for each
        node to decide whether to make any changes to outgoing/incoming edges
        or not. This can speed up processing significantly for large graphs,
        but is more challenging to learn. If set to an integer, the number of
        nodes with edge changes is limited to that integer.

    """
    def __init__(self, num_layers, dim_in, dim_hid, nonlin = torch.nn.ReLU(), filter_edge_edits = False):
        super(GEN, self).__init__()
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError('The number of layers must be a natural number but was %s' % str(num_layers))
        self._num_layers = num_layers
        self._dim_in     = dim_in
        if isinstance(dim_hid, list):
            if len(dim_hid) != num_layers:
                raise ValueError('If a hidden dimensionality for each layer is specified, the number of node dimensionalities must be exactly self._num_layers = %d, but was %d' % (num_layers, len(dim_hid)))
            self._dim_hid = dim_hid
        else:
            self._dim_hid = [dim_hid] * self._num_layers
        # generate the GCN layers
        self._layers = torch.nn.ModuleList()
        # first, an input layer
        self._layers.append(GCN(self._dim_in, self._dim_hid[0]))
        # then, all hidden layers
        for l in range(1, self._num_layers):
            self._layers.append(GCN(self._dim_hid[l-1], self._dim_hid[l]))
        # finally, an output layer for the node edit scores
        self._node = torch.nn.Linear(self._dim_hid[-1], 1)
        self._filter_edge_edits = filter_edge_edits
        if isinstance(filter_edge_edits, int) or filter_edge_edits is True:
            self._edge_filters = torch.nn.Linear(self._dim_hid[-1], 2)
        # and the layers for the edge action scores
        self._edge_u = torch.nn.Linear(self._dim_hid[-1], 1)
        self._edge_v = torch.nn.Linear(self._dim_hid[-1], 1)
        self._edge_w = torch.nn.Linear(1, 1)
        # and the nonlinearity
        self._nonlin = nonlin

    def forward(self, A, X = None):
        """ Computes node and edge edit scores via this graph edit network.

        Parameters
        ----------
        A: class torch.Tensor
            the len(A) x len(A) adjacency matrix.
        X: class torch.Tensor (default = None)
            the len(A) x self._dim_in matrix of initial node representations.


        Returns
        -------
        delta: class torch.Tensor
            a len(A)-dimensional vector of predicted edit scores for each
            node. A negative score means that the node should be deleted,
            a score around zero means no change and a positive score means
            that a new node should be inserted at this node.
        Epsilon: class torch.Tensor
            a len(A) x len(A) matrix of predicted edit scores for each edge.
            A negative score means that the edge should be deleted, a
            score around zero means no change and a positive score means
            that edge egde should be inserted. In other words, A + Epsilon
            should be the new adjacency matrix.
            If self._filter_edge_edits is True, this is a k x l matrix where
            k is the number of positive entries in edge_filters[:, 0] and
            l is the number of positive entries in edge_filters[:, 1].
        edge_filters: class torch.Tensor
            A len(A) x 2 matrix of edge edit filter scores. A negative score
            edge_filters[i, 0] means that no outgoing edge from i should change
            and a negative score edge_filters[i, 1] means that no incoming edge
            to i should change. This is only returned if
            self._filter_edge_edits is True.
            If self._filter_edge_edits is an int and more than
            self._filter_edge_edits are positive, only the top
            self._filter_edge_edits entries should be considered.

        """
        if X is None:
            X = torch.zeros(A.shape[0], self._dim_in)

        A = A.detach()
        # apply each layer
        for l in range(self._num_layers):
            X = self._nonlin(self._layers[l](X, A))
        # apply final layer without nonlinearity
        delta = self._node(X).squeeze(1)
        if self._filter_edge_edits is False:
            # compute edge scores via messages from the input and output node
            # and the inner product of their representations
            in_messages  = self._edge_u(X)
            out_messages = self._edge_v(X).t()
            products     = self._edge_w(torch.mm(X, X.t()).unsqueeze(2)).squeeze(2)
            Epsilon      = in_messages + out_messages + products
            # remove diagonal
            Epsilon      = Epsilon - torch.diag(torch.diag(Epsilon))
            return delta, Epsilon
        else:
            # compute edge filter scores first
            edge_filters = self._edge_filters(X).squeeze(1)
            # identify filtered nodes
            if not isinstance(self._filter_edge_edits, bool):
                if torch.sum(edge_filters[:, 0] > 0.) > self._filter_edge_edits:
                    # if there are too many filtered nodes, limit to the top K
                    out_filtered = torch.argsort(edge_filters[:, 0], descending = True)[:self._filter_edge_edits]
                else:
                    out_filtered = torch.where(edge_filters[:, 0] > 0.)[0]
                if torch.sum(edge_filters[:, 1] > 0.) > self._filter_edge_edits:
                    in_filtered  = torch.argsort(edge_filters[:, 1], descending = True)[:self._filter_edge_edits]
                else:
                    in_filtered = torch.where(edge_filters[:, 1] > 0.)[0]
            else:
                out_filtered = torch.where(edge_filters[:, 0] > 0.)[0]
                in_filtered  = torch.where(edge_filters[:, 1] > 0.)[0]
            # if there are any filtered nodes, compute Epsilon
            if len(out_filtered) > 0 and len(in_filtered) > 0:
                in_messages  = self._edge_u(X[out_filtered, :])
                out_messages = self._edge_v(X[in_filtered, :]).t()
                products     = self._edge_w(torch.mm(X[out_filtered, :], X[in_filtered, :].t()).unsqueeze(2)).squeeze(2)
                Epsilon = in_messages + out_messages + products
                # set diagonal to zero
                Epsilon[out_filtered.unsqueeze(1) == in_filtered.unsqueeze(0)] = 0.
            else:
                # otherwise, set Epsilon to an empty tensor
                Epsilon = torch.tensor([[]])
            return delta, Epsilon, edge_filters

# beta is the 'steepness' factor of the logistic probability distribution.
# For high beta (>= 10), the probability distribution of deletion, replacement,
# and insertion can be very well approximated by just two sigmoids glued
# together. This is the approximation we use here. Accordingly, our loss
# computation is invalid (!) if Beta is set to low values.
_BETA_ = 10

class GEN_loss_crossent(torch.nn.Module):
    """ Implements a variation of the crossentropy loss for edit scores.

    Let v be the vector of node edit scores, Epsilon be the matrix of edge
    edit scores, and e^+ as well as e^- be the vector of outgoing and incoming
    edge filters (if given). Then, the probability distribution for edits is
    defined as follows.

    1. The probability of a deletion of node i is defined as
       1/(1 + exp(10*(v[i]+0.5))), i.e. the logistic distribution with a
       steepness of 10 and an offset of 0.5.
    2. The probability of an insertion at node i is, similarly, defined as
       1/(1 + exp(10*(-v[i]+0.5))), which is again a logistic distribution
       with a steepness of 10 and an offset of 0.5.
    3. Accordingly, the probability of no node action is 1 - the sum of the
       previous two probabilities.

    The probability distribution for edge edits is defined as follows.

    1. The probability of an insertion of edge (i, j) is defined as zero
       if the edge is already present and
       p(+|Epsilon[i, j]) * p(+|e^+[i]) * p(+|e^-[j]) otherwise, where
       p is the standard logistic distributions, i.e.
       p(+|x) = 1 / (1 + exp(-x)). In other words, an edge is only inserted
       if it is not present yet, the edge filters of both the outgoing and
       the incoming edge allow it, and the edge score is high enough.
    2. The probability of a deletion of edge (i, j) is defined as zero
       if the edge is not present and
       (1 - p(+|Epsilon[i, j])) * p(+|e^+[i]) * p(+|e^-[j]) otherwise, where
       p is the standard logistic distributions as before.
       In other words, an edge is only deleted if it is present, if
       the edge filters of both the outgoing and the incoming edge allow it,
       and the edge score is low enough.
    3. Accordingly, the probability of no action on edge (i, j) is 1 - the
       sum of the previous two probabilities.

    Building on these definitions, we define the loss as the crossentropy
    between the desired node and edge edits and the distributions above.
    Note, however, that we do a few approximations to make the computation more
    efficient if edge filters are given. In particular, consider the subset
    of truples (i, j) where i is a node such that no outgoing edge is edited
    and j is a node such that no incoming edge is edited. In that case we
    lower-bound the probability of doing no action to this edge as
    sqrt(p(-|e^+[i]) * p(-|e^-[j])), which yields the crossentropy expression
    -0.5* ( log(p(-|e^+[i])) + log(p(-|e^-[j]))). This can be computed in
    linear time instead of quadratic time, making the loss computation much
    more efficient.

    """
    def __init__(self):
        super(GEN_loss_crossent, self).__init__()

    def forward(self, v, Epsilon, v_true, Epsilon_true, A, edge_filters = None, verbose = False):
        """ Computes the loss.

        Parameters
        ----------
        v: class torch.Tensor
            The predicted node edit scores (m x 1).
        Epsilon: class torch.Tensor
            The predicted edge edit scores (m x m).
        v_true: class torch.True
            The ground truth node edit scores (m x 1).
        Epsilon_true: torch.Tensor
            The ground-truth edge edit scores (m x m).
        A: class torch.Tensor
            The current adjacency matrix (m x m).
        egde_filters: class torch.Tensor (default = None)
            The outgoing and incoming edge filter scores for each node (m x 2).
            If this is given, Epsilon is expected to only have the shape
            K x L, where K and L correspond to the number of positive
            entries in edge_filters[:, 0] and edge_filters[:, 1], respectively.
        verbose: bool (default = False)
            If set to true, status reports are printed.

        Returns
        -------
        loss: class torch.Tensor
            The (approximated) crossentropy loss.

        """
        # Consider the loss for nodes first

        # handle cases that should be deletions
        mask = v_true < -0.5
        # and use the binary crossentropy with logits. This is an approximation
        # because we implicitly assume that the receptive fields for insertion and
        # deletion probabilities are non-overlapping. But this is approximately
        # true for the high-beta regime we're using here.
        node_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            -_BETA_ * (v[mask] + 0.5), torch.ones(len(v[mask])).to(v.device), reduction = 'sum'
        )

        # handle cases that should be insertions
        mask = v_true > +0.5
        # and use the binary crossentropy with logits again.
        node_loss = node_loss + torch.nn.functional.binary_cross_entropy_with_logits(
            -_BETA_ * (-v[mask] + 0.5), torch.ones(len(v[mask])).to(v.device), reduction = 'sum'
        )

        # handle cases that should be no action
        mask = (v_true >= -0.5) * (v_true <= 0.5)
        # in that case we use the binary crossentropy with logits as well, but as
        # logits we consider the absolute value of v.
        node_loss = node_loss + torch.nn.functional.binary_cross_entropy_with_logits(
            -_BETA_ * (torch.abs(v[mask]) - 0.5), torch.ones(len(v[mask])).to(v.device), reduction = 'sum'
        )

        # divide by the number of nodes and beta
        node_loss = node_loss / _BETA_

        if verbose:
            print('node loss: %g' % node_loss.item())

        # now consider the edge loss
        if edge_filters is not None:
            # if edge filters are given, first ensure that we get the filters right
            out_mask = torch.sum(torch.abs(Epsilon_true), 1) > 0.5
            in_mask  = torch.sum(torch.abs(Epsilon_true), 0) > 0.5

            # here we do a binary classification between filtering or not
            filter_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                _BETA_*edge_filters[:, 0], out_mask.double(), reduction = 'sum')
            filter_loss = filter_loss + torch.nn.functional.binary_cross_entropy_with_logits(
                _BETA_*edge_filters[:, 1], in_mask.double(), reduction = 'sum')
            filter_loss = filter_loss * len(A) / _BETA_

            # then reduce A and Epsilon_true to only the submatrix which is
            # left after filtering
            out_mask = edge_filters[:, 0] > 0
            in_mask  = edge_filters[:, 1] > 0
            Epsilon_true = Epsilon_true[out_mask, :][:, in_mask]
            A = A[out_mask, :][:, in_mask]

            if verbose:
                print('filter loss: %g' % filter_loss.item())
        else:
            filter_loss = torch.zeros(1).to(v.device)

        if A.numel() > 0:
            # consider the edges that are present
            mask = A > 0.5
            # here we perform a binary classification between doing nothing and
            # doing an edge deletion
            edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                _BETA_*(-Epsilon[mask]-0.5), (Epsilon_true[mask] < -0.5).double(), reduction = 'sum')

            # consider edges that are not present yet
            mask = A < 0.5
            # here we perform a binary classification between doing nothing and
            # doing an edge insertion
            edge_loss = edge_loss + torch.nn.functional.binary_cross_entropy_with_logits(
                _BETA_*(Epsilon[mask]-0.5), (Epsilon_true[mask] > 0.5).double(), reduction = 'sum')

            # divide by the number of edges and beta
            edge_loss = edge_loss / _BETA_

            if verbose:
                print('edge loss: %g' % edge_loss.item())
        else:
            edge_loss = torch.zeros(1).to(v.device)

        loss = node_loss + filter_loss + edge_loss

        if verbose:
            print('overall loss: %g' % loss.item())

        return loss

class GEN_loss(torch.nn.Module):
    """ Implements a loss similar to the perceptron loss for edit scores.

    The loss for nodes is defined as:
    1. ReLU(delta_pred + 1)² if delta_true is -1,
    2. ReLU(delta_pred - 0.25)² + ReLU(-delta_pred - 0.25)² if delta_true is 0, and
    2. ReLU(-delta_pred + 1)² if delta_true is 1.

    This loss is positive only in three cases:
    1. delta_true is -1 and delta_pred is > -1
    2. delta_true is  0 and delta_pred is < -0.25 or > +0.25
    3. delta_true is +1 and delta_pred is < +1

    In other words, the error is only positive if the action classification
    is wrong or close to be wrong.

    The loss for egdes is defined as:
    ReLU([1 - 2 * A - 2 * Epsilon_true] * Epsilon_pred + abs(Epsilon_true))²

    This loss is positive only in the following cases:
    1. A is 1, Epsilon_true is  0, and Epsilon_pred is <  0
    2. A is 1, Epsilon_true is -1, and Epsilon_pred is > -1
    3. A is 0, Epsilon_true is  0, and Epsilon_pred is >  0
    4. A is 0, Epsilon_true is +1, and Epsilon_pred is < +1

    In other words, the error is only positive if the action classification
    is wrong or margins are violated.

    Optionally, if edge_filters is given, these contribute to the loss as
    well with

    ReLU(epsilon * edge_filters + 0.5) * len(A)

    where epsilon[i] is -1 if sum(Epsilon_true[i, :]) > 0 and +1 otherwise.
    In other words, the loss for edge_filters[i] is positive if edge_filters[i]
    wrongly filters out edge changes that should happen or fails to filter out
    edge changes that shouldn't happen.

    """
    def __init__(self):
        super(GEN_loss, self).__init__()

    def forward(self, delta_pred, Epsilon_pred, delta_true, Epsilon_true, A, edge_filters = None, verbose = False):
        """ Computes the graph edit network loss as specified above.

        Parameters
        ----------
        delta_pred: class torch.Tensor
            The predicted edit scores for all nodes (m x 1).
        Epsilon_pred: class torch.Tensor
            The predicted edit scores for all edges (m x m).
        delta_true: class torch.Tensor
            The desired edit scores for all nodes (m x 1).
        Epsilon_true: class torch.Tensor
            The desired edit scores for all edges (m x m).
        A: class torch.Tensor
            The current adjacency matrix (m x m).
        egde_filters: class torch.Tensor (default = None)
            The edge filter scores for each node (m x 2).

        """
        # compute node loss
        # handle cases that should be deletions
        mask = delta_true < -0.5
        node_loss = torch.sum(torch.pow(torch.nn.functional.relu(delta_pred[mask] + 1.), 2))
        # handle cases where no change should occur
        mask = (delta_true >= -0.5) * (delta_true <= 0.5)
        node_loss = node_loss + torch.sum(torch.pow(torch.nn.functional.relu(delta_pred[mask] -0.25), 2))
        node_loss = node_loss + torch.sum(torch.pow(torch.nn.functional.relu(-delta_pred[mask] -0.25), 2))
        # handle cases that should be insertions
        mask = delta_true > 0.5
        node_loss = node_loss + torch.sum(torch.pow(torch.nn.functional.relu(-delta_pred[mask] + 1.), 2))
        # compute edge loss
        if edge_filters is None:
            edge_loss = torch.sum(torch.pow(torch.nn.functional.relu((1 - 2 * A - 2 * Epsilon_true) * Epsilon_pred + torch.abs(Epsilon_true)), 2))
        else:
            # we first compute the edge filter loss. Note that we weigh that
            # with len(A) to ensure that the loss upper-bounds the loss we may
            # have due to wrong edges
            epsilon_out = 1. - 2. * (torch.sum(torch.abs(Epsilon_true), 1) > 0.5).type(torch.float)
            epsilon_in  = 1. - 2. * (torch.sum(torch.abs(Epsilon_true), 0) > 0.5).type(torch.float)
            edge_loss = torch.sum(
                torch.pow(torch.nn.functional.relu(epsilon_out*edge_filters[:, 0]+0.5), 2)
              + torch.pow(torch.nn.functional.relu(epsilon_in*edge_filters[:, 1]+0.5), 2)) * len(A)
            # then we compute the edge loss for each filtered node
            out_filtered = torch.nonzero(edge_filters[:, 0] > 0.)[:, 0]
            in_filtered  = torch.nonzero(edge_filters[:, 1] > 0.)[:, 0]
            if len(out_filtered) > 0 and len(in_filtered) > 0:
                A = A[out_filtered, :][:, in_filtered]
                Epsilon_true = Epsilon_true[out_filtered, :][:, in_filtered]

                edge_loss = edge_loss + torch.sum(torch.pow(torch.nn.functional.relu((1 - 2 * A - 2 * Epsilon_true) * Epsilon_pred + torch.abs(Epsilon_true)), 2))
        if verbose:
            print('node loss: %g' % node_loss.item())
            print('edge loss: %g' % edge_loss.item())
        return node_loss + edge_loss

def to_edits(A, delta = None, Epsilon = None):
    """ Converts edit scores to edit objects.

    Parameters
    ----------
    A: class torch.Tensor
        the current adjacency matrix.
    delta: class torch.Tensor (default = torch.zeros(len(A)))
        A vector of node edit scores.
    Epsilon: class torch.Tensor (default = torch.zeros(len(A)))
        A matrix of edge edit scores.

    Returns
    -------
    edits: list
        A list of graph edits corresponding to the input scores.

    """
    if delta is None:
        if Epsilon is None:
            return []
        delta = torch.zeros(len(A))
    elif Epsilon is None:
        Epsilon = torch.zeros(len(A), len(A))

    edits = []
    # first, get the set of deleted nodes
    deleted = set()
    for i in range(len(delta)):
        if delta[i] < -0.5:
            deleted.add(i)

    # perform edge edits first because these do not change the index
    # structure
    for i in range(len(Epsilon)):
        if i in deleted:
            continue
        for j in range(len(Epsilon)):
            if j in deleted:
                continue
            if A[i, j] > 0.5 and Epsilon[i, j] < -0.5:
                edits.append(ge.EdgeDeletion(i, j))
            elif A[i, j] < 0.5 and Epsilon[i, j] > 0.5:
                edits.append(ge.EdgeInsertion(i, j))
    # then perform NodeInsertions
    for i in range(len(delta)):
        if delta[i] > 0.5:
            edits.append(ge.NodeInsertion(i, 0.))
    # finally, perform NodeDeletions in descending order
    deleted = list(sorted(deleted))
    for k in range(len(deleted)-1, -1, -1):
        edits.append(ge.NodeDeletion(deleted[k]))
    return edits
