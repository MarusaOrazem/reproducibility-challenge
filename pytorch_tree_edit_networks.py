"""
Implements a simplified version of the graph edit network that works
on tree data and only returns node insertion/deletion scores, but
also a node type classification.

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

import copy
import torch
import edist.ted as ted
import edist.tree_edits as te
import edist.tree_utils as tu
from pytorch_graph_edit_networks import GCN

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TEN(torch.nn.Module):
    """ Computes a node edit score for each node in an input tree and the
    type for any newly created node. In more detail, there are three kinds
    of tree edits: deletions, insertions, and replacements.

    We model deletions via a node edit score of -1, insertions via a score
    of +1, and replacements via a score of 0. In addition, we generate a
    score for each possible node type which we use to determine the type
    of a newly generated node (in case of insertions) or the current node
    (in case of replacements).

    We generate all scores via (shared) graph convolutional layers.

    Attributes
    ----------
    _num_layers: int
        The number of GCN layers.
    _alphabet: list
        The alphabet of possible node types
    _dim_in: int
        len(_alphabet) + dim_in_extra (if given)
    _dim_hid: int or list
        The number of hidden neurons in each layer to compute node
        representations. Can be a list of length _num_layers-1 or a
        single number.
    _nonlin: class torch.nn.Module (default = torch.nn.ReLU())
        The nonlinearity applied after each layer.
    _skip_connections: bool (default = False)
        If set to True, the node representations are computed in a residual
        fashion, i.e. each representation is added to the one in the previous
        layer.
    _dropout: torch.nn.Module (default = torch.nn.Identity)
        if dropout > 0., this is a dropout layer that is applied after graph
        convolutional neural network layer.
    _dim_memory: int (default = 0)
        The number of neurons to store memory information internally during
        a macrostep prediction. This is only used in the predict_macrostep
        and loss_over_time_series functions.

    """
    def __init__(self, num_layers, alphabet, dim_hid, dim_in_extra=0,
                 nonlin=torch.nn.ReLU(), skip_connections=False,
                 dropout=0.0, device=None, dim_memory = 0):
        super(TEN, self).__init__()
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError('The number of layers must be a natural number but was %s' % str(num_layers))
        if not isinstance(alphabet, list):
            raise ValueError('Expected a list of symbols as alphabet argument but was %s' % str(alphabet))
        self._device = device
        self._alphabet = alphabet
        self._alpha_idxs = {}
        for i in range(len(alphabet)):
            self._alpha_idxs[alphabet[i]] = i
        self._num_layers = num_layers
        self._dim_in_extra = dim_in_extra
        self._dim_memory = dim_memory
        self._dim_in     = len(alphabet) + self._dim_in_extra + self._dim_memory
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
        # then, the output layer for edit scores
        self._out_edits = torch.nn.Linear(self._dim_hid[-1], 1)
        # for the insertion location
        self._child_idx_u = torch.nn.Linear(self._dim_hid[-1], 1)
        self._child_idx_v = torch.nn.Linear(self._dim_hid[-1], 1)
        self._child_idx_w = torch.nn.Linear(1, 1)
        # for the child associations during insertions
        self._child_num_u = torch.nn.Linear(self._dim_hid[-1], 1)
        self._child_num_v = torch.nn.Linear(self._dim_hid[-1], 1)
        self._child_num_w = torch.nn.Linear(1, 1)
        # and for the types
        self._out_types = torch.nn.Linear(self._dim_hid[-1], len(alphabet))
        # and the nonlinearity
        self._nonlin = nonlin
        self._skip_connections = skip_connections

        # add a layer to transmit information across time within a macro-step
        if dim_memory > 0:
            self._memory = torch.nn.Linear(self._dim_in, self._dim_memory)

        # dropout
        if dropout < 1E-3:
            self._dropout = torch.nn.Identity()
        else:
            self._dropout = torch.nn.Dropout(p=dropout)

    def forward(self, nodes, adj, X_extra=None, verbose=False, device=None):
        """
        Computes edit scores.

        Parameters
        ----------
        nodes: list
            the node list for the input tree.
        adj: list
            the adjacency list for the input tree.
        X_extra: class torch.Tensor (default = None)
            a len(nodes) x self._dim_in_extra dimensional matrix of node
            attributes.
        verbose: bool (default = False)
            if set to True, debug output is printed.

        Returns
        -------
        delta: class torch.Tensor
            a len(nodes)-dimensional vector of predicted edit scores for each
            node. A negative score means that the node should be deleted,
            a score around zero means no change and a positive score means
            that a new node should be inserted at this node.
        types: class torch.Tensor
            a len(nodes) x len(self._alphabet) matrix of type scores. If
            delta[i] is larger 0.5, the argmax(types[i, :])
            indicates the type for the newly created node. Otherwise it
            indicates the new type for this node.
        Cidx: class torch.Tensor
            a len(nodes) x max_degree+1 matrix of scores to locate an
            insertion. More specifically, if delta[i] is larger than 0.5,
            argmax(Cidx[i, :]) is the index that the newly inserted node will
            have in adj[i].
        Cnum: class torch.Tensor
            a len(nodes) x max_degree matrix of scores to decide which
            children get appended to a newly inserted node. More specifically,
            if delta[i] is larger than 0.5, all consecutive children starting
            from argmax(Cidx[i, :]) with a value larger than 0.5 get appended
            to the newly inserted child.

        """
        if self._dim_in_extra + self._dim_memory > 0 and X_extra.shape[1] != self._dim_in_extra + self._dim_memory:
            raise ValueError('Expected %d extra node features, but got %d' % (self._dim_in_extra + self._dim_memory, X_extra.shape[1]))

        # convert the input tree to a node feature matrix and an adjacency
        # matrix
        X = torch.zeros(len(nodes), len(self._alphabet))
        A = torch.zeros(len(nodes), len(nodes))
        for i in range(len(nodes)):
            c = self._alpha_idxs[nodes[i]]
            X[i, c] = 1.
            for j in adj[i]:
                A[i, j] = 1.
        A = A.detach()
        # concatenate the extra features
        if self._dim_in_extra > 0:
            X = torch.cat((X, X_extra), 1)

        A = A.to(self._device)
        X = X.to(self._device)

        if verbose:
            print('input tree: %s' % tu.tree_to_string(nodes, adj))
            print('initial X')
            print(X.detach().numpy())
            print('initial A')
            print(A.detach().numpy())

        # apply first layer
        H = self._nonlin(self._layers[0](X, A))

        # apply each other layer
        for l in range(1, self._num_layers):
            if self._skip_connections:
                H = H + self._nonlin(self._layers[l](H, A))
            else:
                H = self._nonlin(self._layers[l](H, A))
            H = self._dropout(H)

        if verbose:
            print('final H')
            print(H.detach().numpy())

        # apply score layers without nonlinearity
        delta = self._out_edits(H)
        types = self._out_types(H)
        # for all insertions, compute scores for the children
        max_children = 0
        for i in range(len(adj)):
            if len(adj[i]) > max_children:
                max_children = len(adj[i])
        child_idx_scores = torch.zeros(len(nodes), max_children+1).to(self._device)
        child_idx_scores[:, 0] = -1.
        child_num_scores = torch.zeros(len(nodes), max_children).to(self._device)
        for i in torch.where(delta >= 0.5)[0]:
            # compute the ins/child scores as the edge scores for
            # a graph edit network, i.e. we have a message from the
            # parent, messages from the children, and the products of
            # parent embedding and child embeddings
            if adj[i]:
                parent_message = self._child_idx_u(H[i, :])
                child_messages = self._child_idx_v(H[adj[i], :])
                products = self._child_idx_w(torch.mm(H[adj[i], :], H[i, :].unsqueeze(1)))
                child_idx_scores[i, 1:(len(adj[i])+1)] = (parent_message + child_messages + products).squeeze()
                child_idx_scores[i, :len(adj[i])] = child_idx_scores[i, :len(adj[i])] - (parent_message + child_messages + products).squeeze()

                parent_message = self._child_num_u(H[i, :])
                child_messages = self._child_num_v(H[adj[i], :])
                products = self._child_num_w(torch.mm(H[adj[i], :], H[i, :].unsqueeze(1)))
                child_num_scores[i, :len(adj[i])] = (parent_message + child_messages + products).squeeze()
        return delta, types, child_idx_scores, child_num_scores

    def predict_macrostep(self, nodes, adj, max_microsteps = None, verbose = False):
        """ Predicts the tree edits that should be applied to the given tree.

        In more detail, this performs multiple runs of 'forward', one for
        replacements and deletions, and then additional runs for insertions,
        until no node wishes to perform an insertion anymore.

        Also note that this assumes that self._dim_in_extra is at least the
        maximum degree in the dataset + 1.

        Parameters
        ----------
        nodes: list
            the node list of the input tree.
        adj: list
            the adjacency list of the input tree.
        max_microsteps: int (default = None)
            the maximum number of microsteps before we stop editing. If given,
            this can prevent endless loops.

        Returns
        -------
        edits: list
            A list of edit scripts, one for each micro-step, such that the
            result of applying all the scripts in succession is the output
            tree.
        nodes: list
            The node list of the output tree.
        adj: list
            The adjacency list of the output tree.

        """
        # initialize the outputs
        edits = []

        if verbose:
            print('Received input tree: %s' % tu.tree_to_string(nodes, adj))

        # construct the initial node features
        X = _degree_features(nodes, adj, self._dim_in_extra - 1, 0)
        # expand with zero memory features
        X = torch.cat((X, torch.zeros(len(nodes), self._dim_memory)), dim = 1)
        # predict the changes via the tree edit network
        delta, types, Cidx, Cnum = self(nodes, adj, X)
        # infer the edits corresponding to deletions and replacements
        script_del_rep = te.Script()
        # first add all replacements
        remaining = []
        for i in range(len(nodes)):
            if delta[i] < -0.5:
                continue
            label = self._alphabet[torch.argmax(types[i, :])]
            if label != nodes[i]:
                script_del_rep.append(te.Replacement(i, label))
            remaining.append(i)
        # handle the special case that the root gets deleted, which is only
        # permitted if all children except one get deleted as well
        r = tu.root(adj)
        if delta[r] < -0.5:
            remaining_roots = 0
            stk = [r]
            while stk:
                i = stk.pop()
                for j in adj[i]:
                    if delta[j] < -0.5:
                        stk.append(j)
                    else:
                        remaining_roots += 1
            # if the number of root nodes after deletions is not one, keep the
            # current root
            if remaining_roots != 1:
                delta[r] = 0.
                remaining.append(r)
        # then all deletions, starting from the biggest index to prevent
        # index interference
        for i in range(len(nodes)-1, -1, -1):
            if delta[i] < -0.5:
                script_del_rep.append(te.Deletion(i))

        # prepare memory matrix for next step
        if self._dim_memory > 0:
            X = torch.cat((torch.zeros(len(remaining), len(self._alphabet)), X[remaining, :]), 1)
            for i in range(len(remaining)):
                X[i, self._alpha_idxs[nodes[remaining[i]]]] = 1.
            M = self._nonlin(self._memory(X))

        if verbose:
            print('predicted replacements/deletions: %s' % str(script_del_rep))

        # apply the script
        nodes, adj = script_del_rep.apply(nodes, adj)
        edits.append(script_del_rep)

        # now re-compute the scores and start applying insertions until
        # no marked node performs an insertion anymore.
        ins_marked = None
        while max_microsteps is None or len(edits) < max_microsteps + 1:
            # construct the initial node features
            X = _degree_features(nodes, adj, self._dim_in_extra - 1, 1)
            # concatenate with memory
            if self._dim_memory > 0:
                X = torch.cat((X, M), dim = 1)

#            print('input for ins prediction: %s' % str(X.detach().numpy()))

            # compute edit scores
            delta, types, Cidx, Cnum = self(nodes, adj, X)

            # stop construction once no marked node performs an insertion
            # anymore
            if ins_marked is None:
                if torch.sum(delta > 0.5) < 0.5:
                    break
            elif torch.sum(delta[ins_marked] > 0.5) < 0.5:
                break

#            print('delta for ins prediction: %s' % str(delta.detach().numpy()))
#            print('types for ins prediction: %s' % str(types.detach().numpy()))
#            print('Cidx for ins prediction: %s' % str(Cidx.detach().numpy()))
#            print('Cnum for ins prediction: %s' % str(Cnum.detach().numpy()))

            # initialize the script containing all insertions in the current
            # microstep
            script_ins = te.Script()
            # initialize an index map from indices of the new tree to past
            # indices
            idx_map = list(range(len(nodes)))
            # initialize an array to keep track of index shifts due to inserted
            # nodes
            ins_shift = torch.zeros(len(nodes)+1, dtype=torch.long)
            # check for insertions in marked nodes and construct and index
            # mapping from the former tree to the tree where all insertions
            # have been applied
            if ins_marked is None:
                ins_marked = range(len(nodes))
            for i in ins_marked:
                if delta[i] <= 0.5:
                    continue
                # if we found an insertion, identify the label
                label = self._alphabet[torch.argmax(types[i, :]).item()]
                # and identify the child index
                c = int(torch.argmax(Cidx[i, :len(adj[i])+1]).item())
                # and identify all children to be appended to the new node
                C = 0
                while c + C < len(adj[i]) and Cnum[i, c + C] > 0.5:
                    C += 1
                # append the edit
                script_ins.append(te.Insertion(i + int(ins_shift[i].item()), c, label, C))
                # also identify the index of the new node and adjust the index
                # map accordingly
                if c < len(adj[i]):
                    j = adj[i][c]
                else:
                    j = i
                    while adj[j]:
                        j = adj[j][-1]
                    j += 1
                idx_map.insert(j + ins_shift[j], -1)
                # also update the index shift array
                ins_shift[j:] += 1

            # update marked indices
            ins_marked_new = []
            for i in range(len(idx_map)):
                if idx_map[i] < 0:
                    # we mark an index if it has not existed in the previous tree
                    ins_marked_new.append(i)
                elif idx_map[i] in ins_marked and delta[idx_map[i]] > 0.5:
                    # or if it inserted a node
                    ins_marked_new.append(i)
            ins_marked = ins_marked_new
            # update the memory content
            if self._dim_memory > 0:
                M = torch.zeros(len(idx_map), self._dim_memory)
                for i in range(len(idx_map)):
                    if idx_map[i] < 0:
                        continue
                    x = torch.zeros(self._dim_in)
                    x[self._alpha_idxs[nodes[idx_map[i]]]] = 1.
                    x[len(self._alphabet):] = X[idx_map[i], :]

                    M[i, :] = self._memory(x)

            if verbose:
                print('predicted insertions: %s' % str(script_ins))

            # apply the edits
            nodes, adj = script_ins.apply(nodes, adj)
            edits.append(script_ins)

        return edits, nodes, adj

    def loss_over_time_series(self, trees, verbose = False, custom_alignment = None):
        """ Applies this network to the input time series of trees
        and computes the loss under teacher forcing.

        More precisely, this function first computes the necessary
        edits between one tree and the next via the tree edit distance.
        Then, it first lets the network predict all deletions and
        replacements and checks the loss in that prediction. Next, it
        applies the ground-truth edits and lets the network predict all
        insertions that can directly be applied and computes the loss for
        these predictions. This continues until all predictions are done
        and only the loss for predicting nothing remains. Finally, the
        loss is returned.

        Note that this function computes for each tree as initial feature
        representation a one-hot coding of the child index for each node
        as well as a binary feature indicating whether we currently
        want to predict deletions/replacements or insertions. Accordingly,
        self._dim_in_extra should be the at least the maximum degree in the
        dataset + 1.

        Parameters
        ----------
        trees: list
            A list of trees, each given as a tuple of a node
            and an adjacency list (and possibly additional information,
            which is ignored).
        custom_alignment: function (default = None)
            A default function to compute an alignment between two trees if
            edist.ted.ted_backtrace is not conducive to learning. If not given,
            edist.ted.ted_backtrace is used.

        Returns
        -------
        loss: class torch.Tensor
            A pytorch tensor containing the accumulated loss over the entire
            time series

        """
        # initialize loss as zero
        loss = torch.zeros(1)
        if len(trees) < 2:
            # if there are not at least two trees, we cannot compute a
            # prediction loss
            return loss
        prev_loss = 0.
        # iterate over the time series
        for t in range(1, len(trees)):
            # retrieve the current tree
            nodes, adj = trees[t-1][0], trees[t-1][1]
            # retrieve the next tree
            next_nodes, next_adj = trees[t][0], trees[t][1]

            if verbose:
                print('initial tree in step %d: %s' % (t, tu.tree_to_string(nodes, adj)))
                print('target treee in step %d: %s' % (t, tu.tree_to_string(next_nodes, next_adj)))

            # if both trees are the same, ignore the current step
            if nodes == next_nodes and adj == next_adj:
                continue
            # infer the edits necessary to get from the previous tree
            # to the current tree
            # we use backtracing of the tree edit distance to infer
            # the node alignment
            if custom_alignment is None:
                alignment = ted.ted_backtrace(nodes, adj, next_nodes, next_adj)
            else:
                alignment = custom_alignment(nodes, adj, next_nodes, next_adj)
            # then, we transform this to tree edits
            edits = te.alignment_to_script(alignment, nodes, adj, next_nodes, next_adj)

            # construct the initial node features
            X = _degree_features(nodes, adj, self._dim_in_extra - 1, 0)
            # expand with zero memory features
            if self._dim_memory > 0:
                X = torch.cat((X, torch.zeros(len(nodes), self._dim_memory)), dim = 1)
            # predict the changes via the tree edit network
            delta_pred, types_pred, _, _ = self(nodes, adj, X)
            # initialize matrices for the desired network outputs
            delta = torch.zeros(len(nodes))
            types = torch.zeros(len(nodes), dtype = torch.long)
            for i in range(len(nodes)):
                c = self._alpha_idxs[nodes[i]]
                types[i] = c

            # compute the desired outputs
            tau = 0
            while tau < len(edits):
                edit = edits[tau]
                if isinstance(edit, te.Deletion):
                    delta[edit._index] = -1.
                elif isinstance(edit, te.Replacement):
                    # and set up the desired type
                    types[edit._index] = self._alpha_idxs[edit._label]
                else:
                    break
                tau += 1
            # add a loss for every node where deletion should have been
            # predicted but wasn't
            mask = delta < -0.5
            loss = loss + torch.sum(torch.pow(torch.nn.functional.relu(delta_pred[mask] + 1.), 2))
            # add a loss for every node where replacement should have been
            # predicted but wasn't
            mask = delta >= -0.5
            loss = loss + torch.sum(torch.pow(torch.nn.functional.relu(-delta_pred[mask]), 2))
            # for these nodes, also add a type misclassification loss
            loss = loss + torch.nn.functional.cross_entropy(types_pred[mask, :], types[mask], reduction='sum')

            # prepare memory matrix for next step
            if self._dim_memory > 0:
                remaining = torch.where(mask)[0].tolist()
                X = torch.cat((torch.zeros(len(remaining), len(self._alphabet)), X[remaining, :]), 1)
                for i in range(len(remaining)):
                    X[i, self._alpha_idxs[nodes[remaining[i]]]] = 1.
                M = self._nonlin(self._memory(X))

            # apply all replacements and deletions
            script_del_rep = te.Script(edits[:tau])
            if verbose:
                print('deletion and replacement edits: %s' % str(script_del_rep))
                print('replacement/deletion loss: %g' % (loss.item() - prev_loss))
                prev_loss = loss.item()

            nodes, adj = script_del_rep.apply(nodes, adj)

            # consider the remaining script
            del edits[:tau]
            # now all deletions and replacements are completed. Next, we
            # try to apply as many insertions as possibble

            # continue as long as we still have edits left
            ins_marked = None
            while len(edits) > 0:
                next_edits = []

                # copy the tree to have a representation of the tree after the
                # current round of insertions has been epplied.
                intermediate_nodes = copy.copy(nodes)
                intermediate_adj   = copy.deepcopy(adj)

                if verbose:
                    print('current tree: %s' % tu.tree_to_string(intermediate_nodes, intermediate_adj))
                    print('remaining insertion edits: %s' % edits)

                # and copy it antoher times for an index representation
                index_nodes = copy.copy(nodes)
                index_adj   = copy.deepcopy(adj)
                # set up new arrays to store the desired outputs of the network
                delta = torch.zeros(len(nodes))
                types = torch.zeros(len(nodes), dtype = torch.long)
                for i in range(len(nodes)):
                    c = self._alpha_idxs[nodes[i]]
                    types[i] = c
                Cidx = torch.zeros(len(nodes), dtype = torch.long)
                Cnum = torch.zeros(len(nodes), dtype = torch.long)
                # set up a list to maintain an overview of indices where we did
                # perform insertions
                indices = list(range(len(nodes)))
                # iterate over the remaining edits, which should all be
                # insertions
                for edit in edits:
                    if not isinstance(edit, te.Insertion):
                        raise ValueError('Internal error; expected only insertions after deletions and replacements')
                    if edit._parent_index < 0:
                        raise ValueError('Internal error: Currently, tree edit networks can not deal with insertions at the root. Please ensure that such edits do not happen.')

                    # get the parent index according to the index list before
                    # the insertion
                    p = indices[edit._parent_index]
                    # adjust the indices list
                    if edit._child_index < len(index_adj[edit._parent_index]):
                        j = index_adj[edit._parent_index][edit._child_index]
                    else:
                        orl = edit._parent_index
                        while(index_adj[orl]):
                            orl = index_adj[orl][-1]
                        j = orl + 1
                    indices.insert(j, None)
                    # apply the edit to the index tree
                    edit.apply_in_place(index_nodes, index_adj)

                    # only apply the edit if the parent has existed before or
                    # has not yet performed an edit
                    if p is not None and delta[p] <= 0.5:
                        delta[p] = 1.
                        types[p] = self._alpha_idxs[edit._label]
                        Cidx[p] = edit._child_index
                        Cnum[p] = edit._num_children
                        # re-set the index of this edit and apply it to the intermediate tree
                        edit._parent_index = p + int(torch.sum(delta[:p] > 0.5).item())

                        edit.apply_in_place(intermediate_nodes, intermediate_adj)
                    else:
                        # otherwise, postpone it for the next step
                        next_edits.append(edit)
                # construct the initial node features
                X = _degree_features(nodes, adj, self._dim_in_extra - 1, 1)
                # concatenate with memory
                if self._dim_memory > 0:
                    X = torch.cat((X, M), dim = 1)
                # predict the changes via the tree edit network
                delta_pred, types_pred, Cidx_pred, Cnum_pred = self(nodes, adj, X)

                # constrain our scope only to marked elements
                if ins_marked is not None:

                    delta = delta[ins_marked]
                    delta_pred = delta_pred[ins_marked]
                    types = types[ins_marked]
                    types_pred = types_pred[ins_marked, :]
                    Cidx  = Cidx[ins_marked]
                    Cidx_pred  = Cidx_pred[ins_marked, :]
                    Cnum  = Cnum[ins_marked]
                    Cnum_pred  = Cnum_pred[ins_marked, :]
                else:
                    ins_marked = list(range(len(nodes)))

                # compute the loss for the current insertions

                # punish cases where a replacement should have been predicted
                # but wasn't
                mask = delta <= 0.5
                loss = loss + torch.sum(torch.pow(torch.nn.functional.relu(delta_pred[mask]), 2))
                # punish cases where an insertion should have been predicted
                # but wasn't
                mask = delta > 0.5
                loss = loss + torch.sum(torch.pow(torch.nn.functional.relu(-delta_pred[mask] + 1.), 2))
                # punish type misclassifications
                loss = loss + torch.nn.functional.cross_entropy(types_pred, types, reduction='sum')
                # punish child index misclassifications
                loss = loss + torch.nn.functional.cross_entropy(Cidx_pred[mask, :], Cidx[mask], reduction='sum')
                for i in torch.where(mask)[0]:
                    if Cidx[i] >= len(adj[ins_marked[i]]):
                        continue
                    if Cidx[i] + Cnum[i] > len(Cnum_pred[i, :]) or Cidx[i] + Cnum[i] > len(adj[ins_marked[i]]):
                        raise ValueError('Expected child index %d but had only %d child num scores' % (int((Cidx[i] + Cnum[i]).item()), len(Cnum_pred[i, :])))

                    # punish small child scores for children that should be associated
                    should_be_large = Cnum_pred[i, Cidx[i]:(Cidx[i] + Cnum[i])]
                    loss = loss + torch.sum(torch.pow(torch.nn.functional.relu(-should_be_large + 1.), 2))
                    # punish large child scores that should be small
                    if Cidx[i] + Cnum[i] == len(adj[ins_marked[i]]):
                        continue
                    should_be_small = Cnum_pred[i, Cidx[i]+Cnum[i]]
                    loss = loss + torch.pow(torch.nn.functional.relu(should_be_small), 2)

                if verbose:
                    print('insertion loss: %g' % (loss.item() - prev_loss))
                    prev_loss = loss.item()

                # initialize an index map from indices of the new tree to past
                # indices
                idx_map = list(range(len(nodes)))
                # initialize an array to keep track of index shifts due to inserted
                # nodes
                ins_shift = torch.zeros(len(nodes)+1, dtype=torch.long)
                for i in range(len(ins_marked)):
                    if delta[i] <= 0.5:
                        continue
                    # identify the index of the new node and adjust the index
                    # map accordingly
                    c = Cidx[i]
                    if c < len(adj[ins_marked[i]]):
                        j = adj[ins_marked[i]][c]
                    else:
                        j = ins_marked[i]
                        while adj[j]:
                            j = adj[j][-1]
                        j += 1
                    idx_map.insert(j + ins_shift[j], -1)
                    # also update the index shift array
                    ins_shift[j:] += 1

                # ensure that the index map has the right length
                if len(idx_map) != len(intermediate_nodes):
                    raise ValueError('internal error: index map does not reflect tree size after applying all insertions')

                # update marked indices
                ins_marked_new = []
                for i in range(len(idx_map)):
                    if idx_map[i] < 0:
                        # we mark an index if it has not existed in the previous tree
                        ins_marked_new.append(i)
                    elif idx_map[i] in ins_marked and delta[ins_marked.index(idx_map[i])] > 0.5:
                        # or if it performed an insertion
                        ins_marked_new.append(i)
                ins_marked = ins_marked_new
                # update the memory content
                if self._dim_memory > 0:
                    M = torch.zeros(len(idx_map), self._dim_memory)
                    for i in range(len(idx_map)):
                        if idx_map[i] < 0:
                            continue
                        x = torch.zeros(self._dim_in)
                        x[self._alpha_idxs[nodes[idx_map[i]]]] = 1.
                        x[len(self._alphabet):] = X[idx_map[i], :]

                        M[i, :] = self._memory(x)

                if verbose:
                    print('marked indices: %s' % str(ins_marked))

                # re-set current tree
                nodes, adj = intermediate_nodes, intermediate_adj
                # re-set the edit list
                edits = next_edits

            # ensure that we arrived at the target program
            if nodes != next_nodes or adj != next_adj:
                raise ValueError('Internal error: script did not arrive at target tree. Expected %s but got %s' % (tu.tree_to_string(next_nodes, next_adj), tu.tree_to_string(nodes, adj)))
            # construct the initial node features for the next tree
            X = _degree_features(next_nodes, next_adj, self._dim_in_extra - 1, 1)
            # concatenate with memory
            if self._dim_memory > 0:
                X = torch.cat((X, M), dim = 1)
            # predict the changes via the tree edit network
            delta_pred, types_pred, child_indices_pred, child_numbers_pred = self(next_nodes, next_adj, X)
            if ins_marked is None:
                ins_marked = list(range(len(next_nodes)))
            # ensure that no insertions are predicted anymore
            loss = loss + torch.sum(torch.pow(torch.nn.functional.relu(delta_pred[ins_marked]), 2))
            # and that the types stay as they are
            types = torch.zeros(len(next_nodes), dtype=torch.long)
            for i in ins_marked:
                c = self._alpha_idxs[next_nodes[i]]
                types[i] = c
            loss = loss + torch.nn.functional.cross_entropy(types_pred[ins_marked,:], types[ins_marked], reduction='sum')

            if verbose:
                print('micro-step stop loss: %g' % (loss.item() - prev_loss))
                prev_loss = loss.item()
        return loss / (len(trees) - 1.)


def _degree_features(nodes, adj, max_degree, mode):
    """ Computes a len(nodes) x max_degree + 1 feature matrix containing for
    each node the child index and, in the last column, the given mode
    mode indicator.

    Parameters
    ----------
    nodes: list
        A node list.
    adj: list
        An adjacency list.
    max_degree: int
        The maximum degree expected. If for any node i,
        len(adj[i]) >  max_degree, this will cause an error.
    mode: float
        A single number indicating a mode feature for all nodes.

    Returns
    -------
    X: class torch.Tensor
        A len(nodes) x max_degree + 1 feature matrix containing for
        each node the child index and, in the last column, the given mode
        mode indicator.

    """
    X = torch.zeros(len(nodes), max_degree + 1)
    for i in range(len(adj)):
        if len(adj[i]) > max_degree:
            raise ValueError('Adjacency list of tree %s at node %d exceeds given maximum degree %d' % (tu.tree_to_string(nodes, adj), i, max_degree))
        for c in range(len(adj[i])):
            X[adj[i][c], c] = 1.
    X[:, -1] = mode
    return X


def to_edits(nodes, adj, delta, types, child_idx_scores, child_num_scores, alphabet, adjust_indices = True):
    """ Converts edit scores to edit objects.

    Parameters
    ----------
    nodes: list
        the current trees node list.
    adj: list
        the current tree adjacency list.
    delta: class torch.Tensor
        A len(nodes) vector of node edit scores.
    types:  class torch.Tensor
        A len(nodex) x len(alphabet) matrix of edge edit scores.
    Cidx: class torch.Tensor
        A len(nodes) x max_degree + 1 matrix of child location scores for
        insertions.
    Cnum: class Torch.Tensor
        A len(nodes) x max_degree matrix of child association scores for
        insertions.
    alphabet: list
        The node alphabet.
    adjust_indices: bool (default = True)
        If set to True, adjusts the indices of insertions such that the
        returned script can be applied as is. If set to False, indices are
        not adjusted, but every returned edit is valid individually.

    Returns
    -------
    script: class edist.tree_edits.Script
        A script of tree edits corresponding to the input scores.

    """
    if len(nodes) != len(delta):
        raise ValueError('Inputs were inconsistent; received %d nodes but %d node edit scores' % (len(nodes), len(delta)))
    if len(nodes) != len(types):
        raise ValueError('Inputs were inconsistent; received %d nodes but %d type scores' % (len(nodes), len(delta)))

    edits = []
    # perform replacements first because we don't change the tree structure
    # for that
    for i in range(len(nodes)):
        if abs(delta[i].item()) <= 0.5:
            label = torch.argmax(types[i, :]).item()
            label = alphabet[label]
            if label == nodes[i]:
                continue
            edits.append(te.Replacement(i, label))

    # perform node deletions next, starting from the last one to prevent
    # index interference
    dels = set()
    for i in range(len(nodes)-1, -1, -1):
        if delta[i] < -0.5:
            edits.append(te.Deletion(i))
            dels.add(i)
    if adjust_indices:
        # create an array containing the accumulated index shift due to deletions
        del_shift = torch.zeros(len(nodes), dtype=torch.long)
        if 0 in dels:
            del_shift[0] = -1
        for i in range(1, len(nodes)):
            del_shift[i] = del_shift[i-1]
            if i in dels:
                del_shift[i] -= 1
        # and one for the shift due to insertions
        ins_shift = torch.zeros(len(nodes)+1, dtype=torch.long)
        # and compute the number of non-deleted nodes that are children of
        # each node in the tree
        Cs = []
        _num_descendants(adj, dels, 0, Cs)
    # then perform node insertions, starting with the first one to prevent
    # interference with child indices
    for i in range(len(nodes)):
        i_shifted = i
        if adjust_indices:
            # adjust the index based on previous deletions and insertions
            if i > 0:
                ins_shift[i] += ins_shift[i-1]
            i_shifted += int((ins_shift[i] + del_shift[i]).item())

        if delta[i] > +0.5:
            # identify the label
            label = torch.argmax(types[i, :]).item()
            label = alphabet[label]

            # compute the children left after deletions
            cs    = []
            if adjust_indices:
                for c in range(len(adj[i])):
                    if adj[i][c] in dels:
                        continue
                    cs.append(c)
            else:
                cs = list(range(len(adj[i])))
            # if there are no children, append an edit right away and
            # continue
            if len(cs) == 0:
                edits.append(te.Insertion(i_shifted, 0, label, 0))
                if adjust_indices:
                    ins_shift[i+1] += 1
                continue
            # otherwise, compute the index of the child where we perform the
            # insertion
            child_idx = 0
            score = child_idx_scores[i, 0]
            for c2 in range(len(cs)):
                c = cs[c2]
                if child_idx_scores[i, c+1] > score:
                    child_idx = c2+1
            # and infer the number of children
            num_children = 0
            for c2 in range(child_idx, len(cs)):
                if child_num_scores[i, cs[c2]] > 0.5:
                    num_children += 1
                else:
                    break
            # adjust the shift index of children
            if adjust_indices:
                if child_idx < len(cs):
                    ins_shift[adj[i][cs[child_idx]]] += 1
                else:
                    # if the child index is too large for the
                    # adjacency list, compute the outermost right leaf
                    # of i
                    orl = i
                    cs_orl = cs
                    while cs_orl:
                        orl = adj[orl][cs_orl[-1]]
                        # re-compute cs
                        cs_orl = []
                        for c in range(len(adj[orl])):
                            if adj[orl][c] in dels:
                                continue
                            cs_orl.append(c)
                    ins_shift[orl+1] += 1
                # shift the child index as well
                c = 0
                cs.append(len(adj[i]))
                C = child_idx
                # iterate over all deleted children and add their
                # number of non-deleted children to the child index
                c_last = -1
                for c2 in range(C+1):
                    for c in range(c_last+1, cs[c2]):
                        child_idx += Cs[adj[i][c]]
                    c_last = cs[c2]
            edits.append(te.Insertion(i_shifted, child_idx, label, num_children))

    return te.Script(edits)

def _num_descendants(adj, dels, i, out):
    # add a counting variable for the number of descendants for the current node
    out.append(0)
    # iterate over all children of the current node
    for j in adj[i]:
        # call this method recursively for the child
        _num_descendants(adj, dels, j, out)
        # check if the current child is deleted
        if j in dels:
            # if it is, add the number of descendants of the child
            out[i] += out[j]
        else:
            # otherwise, add only 1
            out[i] += 1
