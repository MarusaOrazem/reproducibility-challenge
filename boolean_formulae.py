"""
Generates data for the 'boolean formulae' data set, where we start with a
random Boolean formula and then apply all of the following simplification
rules until none applies anymore:

1. and(x, False) is equivalent to False.
2. and(x, True) is equivalent to x.
3. or(x, True) is equivalent to True.
4. or(x, False) is equivalent to x.
5. and(x, x) is equivalent to x.
6. or(x, x) is equivalent to x.
7. and(x, not_x) is equivalent to False.
8. or(x, not_x) is equivalent to True.

"""

# Copyright (C) 2020-2021
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
import numpy as np
from edist.alignment import Alignment
import edist.tree_edits as tree_edits
import edist.tree_utils as tree_utils
import pytorch_tree_edit_networks as ten
import peano_addition

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

alphabet = ['and', 'or', 'x', 'y', 'not_x', 'not_y', 'True', 'False', 'root']

def generate_time_series(max_op = 3):
    """ Generates a random Boolean formula with at most max_bin_op binary
    operators and then applies the eight simplification rules listed above
    until none of them applies anymore.

    For more details, refer to the _generate_tree() and _simplify() method
    respectively.

    Parameters
    ----------
    max_op: int (default = 3)
        The maximum number of 'and'/'or' operators in the generated tree. Note
        that the space of possible trees grows exponentially in this parameter.
        For the default value, already ~30k trees are possible.

    Returns
    -------
    time_series: list
        A list of trees with successively simpler versions of the initial
        Boolean formula, until none of the above listed thirteen simplification
        rules is applicable anymore.

    """
    # generate a tree first
    nodes, adj = _generate_tree(max_op)
    # and simplify it
    try:
        return _simplify(nodes, adj)
    except Exception as ex:
        print(tree_utils.tree_to_string(nodes, adj, indent = True, with_indices = True))
        raise ex

p_op = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]
p_non_op = [0., 0., 0.25, 0.25, 0.25, 0.25]

def _generate_tree(max_op = 3):
    """ Generates a random Boolean formula with at most max_bin_op binary
    operators.

    In more detail, the generation is done via a stochastic regular tree
    grammar with probability 0.3 for 'and'/'or' respectively, and probability
    0.1 for 'x', 'y', 'not_x', and 'not_y' respectively. Any operator receives
    precisely two children. If all operators have been used, the probabilities
    are 0.25 for 'x', 'y', 'not_x', and 'not_y' respectively.

    Parameters
    ----------
    max_op: int (default = 3)
        The maximum number of 'and'/'or' operators in the generated tree. Note
        that the space of possible trees grows exponentially in this parameter.
        For the default value, already 10788 trees are possible.

    Returns
    -------
    nodes: list
        The node list of the generated tree.
    adj: list
        The adjacency list of the generated tree.

    """

    # initialize node and adjacency list
    nodes = ['root']
    adj = [[]]

    # initialize a stack for generation which always contains the parent index
    stk = [0]
    while stk:
        # pop the current parent from the stack
        p = stk.pop()
        # sample a label for the new node with a probability distribution
        # dependent on the remaining numbr of operations
        if max_op > 0:
            r = np.random.choice(len(p_op), 1, p = p_op)
        else:
            r = np.random.choice(len(p_non_op), 1, p = p_non_op)
        # append the new node to the tree
        i = len(nodes)
        label = alphabet[r[0]]
        nodes.append(label)
        adj.append([])
        adj[p].append(i)
        # push new entries on the stack, depending on the label
        if label in ['and', 'or']:
            stk.append(i)
            stk.append(i)
            max_op -= 1
    # return the generated tree
    return nodes, adj

def _simplify(nodes, adj, verbose = False):
    """ Applies the eight simplification rules listed above to the given
    tree until none applies anymore and constructs a time series out of all
    intermediate states.

    Attributes
    ----------
    nodes: list
        The node list of the tree to be simplified.
    adj: list
        The adjacency list of the tree to be simplified.

    Returns
    -------
    time_series: list
        A list of trees with successively simpler versions of the initial
        given tree, until none of the above listed thirteen simplification
        rules is applicable anymore.

    """
    # initialize the time series
    time_series = [(nodes, adj)]
    while True:

        if verbose:
            print('Current tree: %s' % tree_utils.tree_to_string(nodes, adj))

        # iterate over the tree and aggregate indices that should be deleted
        # as well as replacement edits
        script = tree_edits.Script()
        to_be_deleted = set()
        for i in range(len(nodes)):
            children = []
            for j in adj[i]:
                children.append(nodes[j])

            if nodes[i] == 'and' and 'False' in children:
                # first rule: and(x, False) is equivalent to False.
                if 'and' in children or 'or' in children:
                    # If 'x' is, in fact, an operator, only delete the operator
                    # first and not its children; otherwise, this will be
                    # hard to learn
                    for c in range(len(children)):
                        if children[c] == 'False':
                            continue
                        to_be_deleted.add(adj[i][c])
                else:
                    # delete all children but one false
                    first_false = children.index('False')
                    for c in range(len(children)):
                        if c == first_false:
                            continue
                        to_be_deleted.add(adj[i][c])
                    # and delete the parent operator
                    to_be_deleted.add(i)
                if verbose:
                    print('first rule applies at node %d, yielding deletions %s' % (i, str(to_be_deleted)))
            elif nodes[i] == 'and' and 'True' in children:
                # second rule: and(x, True) is equivalent to x.
                # apply the simplifying edits, i.e. delete 'and' as well as
                # 'True'
                to_be_deleted.add(i)
                for c in range(len(children)):
                    if children[c] == 'True':
                        to_be_deleted.add(adj[i][c])
                if verbose:
                    print('second rule applies at node %d, yielding deletions %s' % (i, str(to_be_deleted)))
            elif nodes[i] == 'or' and 'True' in children:
                # third rule: or(x, True) is equivalent to True.
                if 'and' in children or 'or' in children:
                    # If 'x' is, in fact, an operator, only delete the operator
                    # first and not its children; otherwise, this will be
                    # hard to learn
                    for c in range(len(children)):
                        if children[c] == 'True':
                            continue
                        to_be_deleted.add(adj[i][c])
                else:
                    # delete all children but one True
                    first_true = children.index('True')
                    for c in range(len(children)):
                        if c == first_true:
                            continue
                        to_be_deleted.add(adj[i][c])
                    # and delete the parent operator
                    to_be_deleted.add(i)
                if verbose:
                    print('third rule applies at node %d, yielding deletions %s' % (i, str(to_be_deleted)))
            elif nodes[i] == 'or' and 'False' in children:
                # fourth rule: or(x, False) is equivalent to x.
                # apply the simplifying edits, i.e. delete 'or' as well as
                # 'False'
                to_be_deleted.add(i)
                for c in range(len(children)):
                    if children[c] == 'False':
                        to_be_deleted.add(adj[i][c])
                to_be_deleted.add(i)
                if verbose:
                    print('fourth rule applies at node %d, yielding deletions %s' % (i, str(to_be_deleted)))
            elif nodes[i] in ['and' ,'or'] and children[0] in ['x', 'y'] and children.count(children[0]) == len(children):
                # fifth/sixth rule: and/or(x, x) is equivalent to x.
                # apply the simplifying edits, i.e. delete the 'and' and all
                # children but one
                to_be_deleted.add(i)
                for c in range(len(children)-1):
                    to_be_deleted.add(adj[i][c])
                if verbose:
                    print('rule five or six applies at node %d, yielding deletions %s' % (i, str(to_be_deleted)))
            elif nodes[i] in ['and', 'or'] and (('not_x' in children and 'x' in children) or ('not_y' in children and 'y' in children)):
                # seventh/eighth rule: and/or(x, not(x)) is equivalent to
                # False/True.
                if 'and' in children or 'or' in children:
                    # if we have a binary operator in the children, first only delete
                    # everything but the relevant variables
                    if 'not_x' in children and 'x' in children:
                        relevant_children = ['not_x', 'x']
                    else:
                        relevant_children = ['not_y', 'y']
                    for c in range(len(children)):
                        if children[c] in relevant_children:
                            continue
                        to_be_deleted.add(adj[i][c])
                    if verbose:
                        print('rule seven or eight applies at node %d, yielding deletions %s' % (i, str(to_be_deleted)))
                else:
                    # apply the simplifying edits, i.e. delete everything and
                    # replace the root with False/True
                    new_root_label = str(nodes[i] == 'or')
                    script.append(tree_edits.Replacement(i, new_root_label))
                    for j in adj[i]:
                        to_be_deleted.add(j)
                    if verbose:
                        print('rule seven or eight applies at node %d, yielding deletions %s and rep(%d, %s)' % (i, str(to_be_deleted), i, new_root_label))
        # add deletions
        to_be_deleted = list(sorted(to_be_deleted, reverse=True))
        for i in to_be_deleted:
            script.append(tree_edits.Deletion(i))
        # check if we have changed anything this iteration
        if len(script) == 0:
            # if not, end the process
            break
        # otherwise, append a new entry to the time series and continue
        nodes, adj = script.apply(nodes, adj)
        time_series.append((nodes, adj))
    return time_series


def compute_loss(model, time_series, verbose = False):
    """ A custom loss function for the Boolean addition task using a protocol
    with only a single predictive step between graphs.

    Parameters
    ----------
    model: class pytorch_tree_edit_networks.TEN
        A tree edit network for which the loss shall be computed.
    time_series: list
        A list of trees as returned by _simplify.

    Returns
    -------
    loss: torch.tensor
        The graph edit network loss between the tree edit network predictions
        and the scores that ought to be generated.

    """
    # verify that the model does not expect memory
    if model._dim_memory > 0:
        raise ValueError('The boolean_formulae.compute_loss function is not compatible with a tree edit network with memory.')
    # initialize loss
    loss = torch.zeros(1)
    for t in range(len(time_series)):
        nodes, adj = time_series[t]
        # retrieve the parent of each node
        pi = np.zeros(len(nodes), dtype=int)
        for i in range(len(nodes)):
            for j in adj[i]:
                pi[j] = i
        # construct the initial node features for the current tree
        X = ten._degree_features(nodes, adj, model._dim_in_extra - 1, 0)
        # perform the prediction of the tree edit network
        delta_pred, types_pred, _, _ = model(nodes, adj, X)
        # initialize desired outputs
        delta = torch.zeros(len(nodes))
        types = torch.zeros(len(nodes), dtype=torch.long)
        # initializes types with the same type as before
        for i in range(len(nodes)):
            types[i] = alphabet.index(nodes[i])
        # iterate over the tree and aggregate deletions as well as replacements
        for i in range(len(nodes)):
            children = []
            for j in adj[i]:
                children.append(nodes[j])

            if nodes[i] == 'and' and 'False' in children:
                # first rule: and(x, False) is equivalent to False.
                if 'and' in children or 'or' in children:
                    # If 'x' is, in fact, an operator, only delete the operator
                    # first and not its children; otherwise, this will be
                    # hard to learn
                    for c in range(len(children)):
                        if children[c] == 'False':
                            continue
                        delta[adj[i][c]] = -1.
                else:
                    # delete all children but one false
                    first_false = children.index('False')
                    for c in range(len(children)):
                        if c == first_false:
                            continue
                        delta[adj[i][c]] = -1.
                    # and delete the parent operator
                    delta[i] = -1.
            elif nodes[i] == 'and' and 'True' in children:
                # second rule: and(x, True) is equivalent to x.
                # apply the simplifying edits, i.e. delete 'and' as well as
                # 'True'
                delta[i] = -1.
                for c in range(len(children)):
                    if children[c] == 'True':
                        delta[adj[i][c]] = -1.
            elif nodes[i] == 'or' and 'True' in children:
                # third rule: or(x, True) is equivalent to True.
                if 'and' in children or 'or' in children:
                    # If 'x' is, in fact, an operator, only delete the operator
                    # first and not its children; otherwise, this will be
                    # hard to learn
                    for c in range(len(children)):
                        if children[c] == 'True':
                            continue
                        delta[adj[i][c]] = -1.
                else:
                    # delete all children but one True
                    first_true = children.index('True')
                    for c in range(len(children)):
                        if c == first_true:
                            continue
                        delta[adj[i][c]] = -1.
                    # and delete the parent operator
                delta[i] = -1.
            elif nodes[i] == 'or' and 'False' in children:
                # fourth rule: or(x, False) is equivalent to x.
                # apply the simplifying edits, i.e. delete 'or' as well as
                # 'False'
                delta[i] = -1.
                for c in range(len(children)):
                    if children[c] == 'False':
                        delta[adj[i][c]] = -1.
                delta[i] = -1.
            elif nodes[i] in ['and' ,'or'] and children[0] in ['x', 'y'] and children.count(children[0]) == len(children):
                # fifth/sixth rule: and/or(x, x) is equivalent to x.
                # apply the simplifying edits, i.e. delete the 'and' and all
                # children but one
                delta[i] = -1.
                for c in range(len(children)-1):
                    delta[adj[i][c]] = -1.
            elif nodes[i] in ['and', 'or'] and (('not_x' in children and 'x' in children) or ('not_y' in children and 'y' in children)):
                # seventh/eighth rule: and/or(x, not(x)) is equivalent to
                # False/True.
                if 'and' in children or 'or' in children:
                    # if we have a binary operator in the children, first only delete
                    # everything but the relevant variables
                    if 'not_x' in children and 'x' in children:
                        relevant_children = ['not_x', 'x']
                    else:
                        relevant_children = ['not_y', 'y']
                    for c in range(len(children)):
                        if children[c] in relevant_children:
                            continue
                        delta[adj[i][c]] = -1.
                else:
                    # apply the simplifying edits, i.e. delete everything and
                    # replace the root with False/True
                    new_root_label = str(nodes[i] == 'or')
                    types[i] = alphabet.index(new_root_label)
                    for j in adj[i]:
                        delta[j] = -1.

        # compute the tree edit network loss, i.e. punish large scores if
        # we want deletions
        mask = delta < -0.5
        if torch.any(mask):
            loss = loss + torch.sum(torch.pow(torch.nn.functional.relu(delta_pred[mask] + 1.), 2))
            if verbose:
                print('deletion loss: %g' % loss.item())
                last_loss = loss.item()
        # punish scores that are large in absolute value if we want replacements
        mask = torch.abs(delta) < 0.5
        if torch.any(mask):
            loss = loss + torch.sum(torch.pow(torch.nn.functional.relu(torch.abs(delta_pred[mask]) - .25), 2))
            # and punish type errors for replacements as well
            loss = loss + torch.nn.functional.cross_entropy(types_pred[mask, :], types[mask], reduction='sum')
            if verbose:
                print('replacement loss: %g' % (loss.item() - last_loss))
                last_loss = loss.item()

    # return loss
    return loss


def predict_step(model, nodes, adj, alpha = None, verbose = False):
    """ A custom prediction function for tree edit networks to perform a
    single-step prediction on a given tree.

    Parameters
    ----------
    model: class pytorch_tree_edit_networks.TEN
        A tree edit network for which the prediction shall be computed.
    nodes: list
        the node list of the input tree.
    adj: list
        the adjacency list of the input tree.
    alpha: list (default = None)
        a custom alphabet. The Boolean formulae alphabet per default.
    verbose: bool (default = False)
        if set to True, prints diagnostic information.

    Returns
    -------
    script: class edist.tree_edits.Script
        An edit script which yields the output tree.
    nodes: list
        The node list of the output tree.
    adj: list
        The adjacency list of the output tree.

    """
    if alpha is None:
        alpha = alphabet
    return peano_addition.predict_step(model, nodes, adj, alpha, verbose)

def boolean_alignment(nodes, adj, next_nodes, next_adj):
    """ A custom alignment function between a tree and its successor according
    to _simplify. We need this function because the default alignments returned
    by edist.ted are needlessly hard to learn.

    Parameters
    ----------
    nodes: list
        The node list of the tree to be simplified.
    adj: list
        The adjacency list of the tree to be simplified.
    next_nodes: list
        The node list of the simplified tree.
    next_adj: list
        The adjacency list of the simplified tree.

    Returns
    -------
    alignment: class edist.alignment.Alignment
        The alignment between nodes and next_nodes.

    """
    # note all deleted nodes
    to_be_deleted = set()
    # iterate over the tree and look for remaining addition operators
    for i in range(len(nodes)):
        children = []
        for j in adj[i]:
            children.append(nodes[j])

        if nodes[i] == 'and' and 'False' in children:
            # first rule: and(x, False) is equivalent to False.
            if 'and' in children or 'or' in children:
                # If 'x' is, in fact, an operator, only delete the operator
                # first and not its children; otherwise, this will be
                # hard to learn
                for c in range(len(children)):
                    if children[c] == 'False':
                        continue
                    to_be_deleted.add(adj[i][c])
            else:
                # delete all children but one false
                first_false = children.index('False')
                for c in range(len(children)):
                    if c == first_false:
                        continue
                    to_be_deleted.add(adj[i][c])
                # and delete the parent operator
                to_be_deleted.add(i)
        elif nodes[i] == 'and' and 'True' in children:
            # second rule: and(x, True) is equivalent to x.
            # apply the simplifying edits, i.e. delete 'and' as well as
            # 'True'
            to_be_deleted.add(i)
            for c in range(len(children)):
                if children[c] == 'True':
                    to_be_deleted.add(adj[i][c])
        elif nodes[i] == 'or' and 'True' in children:
            # third rule: or(x, True) is equivalent to True.
            if 'and' in children or 'or' in children:
                # If 'x' is, in fact, an operator, only delete the operator
                # first and not its children; otherwise, this will be
                # hard to learn
                for c in range(len(children)):
                    if children[c] == 'True':
                        continue
                    to_be_deleted.add(adj[i][c])
            else:
                # delete all children but one True
                first_true = children.index('True')
                for c in range(len(children)):
                    if c == first_true:
                        continue
                    to_be_deleted.add(adj[i][c])
                # and delete the parent operator
                to_be_deleted.add(i)
        elif nodes[i] == 'or' and 'False' in children:
            # fourth rule: or(x, False) is equivalent to x.
            # apply the simplifying edits, i.e. delete 'or' as well as
            # 'False'
            to_be_deleted.add(i)
            for c in range(len(children)):
                if children[c] == 'False':
                    to_be_deleted.add(adj[i][c])
            false_idx = adj[i][0] if nodes[adj[i][0]] == 'False' else adj[i][1]
            to_be_deleted.add(false_idx)
            to_be_deleted.add(i)
        elif nodes[i] in ['and' ,'or'] and children[0] in ['x', 'y'] and children.count(children[0]) == len(children):
            # fifth/sixth rule: and/or(x, x) is equivalent to x.
            # apply the simplifying edits, i.e. delete the 'and' and all
            # children but one
            to_be_deleted.add(i)
            for c in range(len(children)-1):
                to_be_deleted.add(adj[i][c])
        elif nodes[i] in ['and', 'or'] and (('not_x' in children and 'x' in children) or ('not_y' in children and 'y' in children)):
            # seventh/eighth rule: and/or(x, not(x)) is equivalent to
            # False/True.
            if 'and' in children or 'or' in children:
                # if we have a binary operator in the children, first only delete
                # everything but the relevant variables
                if 'not_x' in children and 'x' in children:
                    relevant_children = ['not_x', 'x']
                else:
                    relevant_children = ['not_y', 'y']
                for c in range(len(children)):
                    if children[c] in relevant_children:
                        continue
                    to_be_deleted.add(adj[i][c])
            else:
                # apply the simplifying edits, i.e. delete everything and
                # replace the root with False/True
                for j in adj[i]:
                    to_be_deleted.add(j)
    # build the alignment
    alignment = Alignment()
    i, j = 0, 0
    while i < len(nodes):
        if i in to_be_deleted:
            alignment.append_tuple(i, -1)
        else:
            alignment.append_tuple(i, j)
            j += 1
        i += 1
    # return
    return alignment
