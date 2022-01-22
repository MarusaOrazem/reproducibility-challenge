#!/usr/bin/python3
"""
Tests the tree edit network implementation.

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

import unittest
import math
import random
import torch
import edist.tree_edits as te
import edist.tree_utils as tu
import pytorch_tree_edit_networks as ten

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class DynSysTEN(ten.TEN):
    """ This special tree edit network is not learned but pre-set to
    always generate a simple dynamical system over trees, where each 'a' is
    replaced with the motif 'a(b, b)' and each 'b' is replaced with an 'a'.

    """
    def __init__(self):
        super(DynSysTEN, self).__init__(num_layers = 1, alphabet = ['a', 'b'], dim_hid = 1, dim_in_extra = 3, dim_memory = 1)
        self._memory.weight[:, :] = 0.
        self._memory.weight[0, 1] = 1.
        self._memory.weight[0, -1] = 1.
        self._memory.bias[:] = 0.

    def forward(self, nodes, adj, X):
        # set up output arrays
        delta = torch.zeros(len(nodes))
        types = torch.zeros(len(nodes), 2)

        max_degree = 0
        for i in range(len(adj)):
            if len(adj[i]) > max_degree:
                max_degree = len(adj[i])

        Cidx = torch.zeros(len(nodes), max_degree+1)
        Cnum = torch.zeros(len(nodes), max_degree)

        # fill output arrays
        if torch.sum(X[:, -2]) < 0.5:
            mode = 'del/rep'
        else:
            mode = 'ins'
        if mode == 'del/rep':
            # if we are in deletion/replacement mode, we replace every
            # b with an a
            types[:, 0] = 100.
        else:
            # otherwise we insert a new b child for all 'a's that do not 
            # already have two children and do not memorize a 'b'
            for i in range(len(nodes)):
                # leave all 'b's as they are
                if nodes[i] == 'b':
                    types[i, 1] = 100.
                    continue
                # as well as all 'a's with two children (or more), or
                # 'a's which have just been replaced
                elif nodes[i] == 'a' and (X[i, -1] > 0.5 or len(adj[i]) >= 2):
                    types[i, 0] = 100.
                    continue
                # add an insertion for each a with 0 or 1 children
                delta[i] = 1.
                types[i, 1] = 100.
                Cidx[i, len(adj[i])] = 100.
        return delta, types, Cidx, Cnum

class TestTENs(unittest.TestCase):

    def assertClose(self, expected, actual, threshold = 1E-3):
        self.assertTrue(abs(expected - actual) < 1E-3, "expected %g but got %g" % (expected, actual))

    def test_loss_over_time_series(self):
        # set up a time series of just two trees, which needs several edit
        # steps of a tree edit network.
        x_nodes = ['a', 'd']
        x_adj   = [[1], []]
        y_nodes = ['a', 'b', 'c', 'd']
        y_adj   = [[1, 3], [2], [], []]

        alphabet = ['a', 'b', 'c', 'd']

        # set up a very simple model which predicts no edits
        model = ten.TEN(1, alphabet, len(alphabet), 3)
        type_factor = 100.
        # set all weights to zero
        for key in model.state_dict():
            tensor = model.state_dict()[key]
            if len(tensor.shape) > 1:
                tensor[:, :] = 0.
            else:
                tensor[:] = 0.
        # store only the type information in the hidden layer
        model.state_dict()['_layers.0._W.weight'][:,:len(alphabet)] = torch.eye(len(alphabet))
        # map current type information to next-state type information
        model.state_dict()['_out_types.weight'][:,:] = torch.eye(len(alphabet)) * type_factor
        # ensure that the network predicts no change for the first input tree
        edits, nodes, adj = model.predict_macrostep(x_nodes, x_adj)
        self.assertEqual(x_nodes, nodes)
        self.assertEqual(x_adj, adj)
        self.assertEqual([[]], edits)

        # accordingly, the loss for getting from a tree to itself should be zero
        loss = model.loss_over_time_series([(x_nodes, x_adj), (x_nodes, x_adj)])
        self.assertClose(0., loss)

        # compute the loss of this model for getting from x to y
        actual_loss = model.loss_over_time_series([(x_nodes, x_adj), (y_nodes, y_adj)])
        # compare to the expected loss. We expect that we need to insert
        # both b and c, which the network doesn't do, and that the predicted
        # types need to be b and c, which the network doesn't do either.
        # Further, the script needs to be cut into two pieces, because we can
        # not insert both b and c in one step. In Each step, we get one
        # insertion wrong and one type wrong. The insertion loss is 1, the
        # type loss is log(exp(0) / (len(alphabet) - 1 + exp(type_factor)))
        # which is approximately equal to type_factor for large enough values.
        # Finally, we get the child index slightly wrong because we do not
        # clearly enough vote for the first index as insertion location. The
        # according loss is -log(exp(-1) / (max_degree * exp(0) + exp(-1)))
        # = 1 + log(max_degree + exp(-1))
        expected_loss = 1. + type_factor + 1. + math.log(1 + math.exp(-1)) + \
                        1. + type_factor + 1. + math.log(2 + math.exp(-1))
        self.assertClose(expected_loss, actual_loss)

        # next, compare to the desired output in the dynamical system
        x_nodes = ['b']
        x_adj   = [[]]
        y_nodes = ['a']
        y_adj   = [[]]
        z_nodes = ['a', 'b', 'b']
        z_adj   = [[1, 2], [], []]

        actual_loss = model.loss_over_time_series([(x_nodes, x_adj), (y_nodes, y_adj), (z_nodes, z_adj)])
        # the expected loss is getting the type wrong for the replacement and
        # then getting both insertions with their types and an insertion
        # location wrong
        expected_loss = (type_factor + 1. + type_factor + 1 + type_factor + math.log(1 + math.exp(-1))) / 2
        self.assertClose(expected_loss, actual_loss)

        # however, if we use the DynSysTEN model, our loss should be zero
        model = DynSysTEN()
        actual_loss = model.loss_over_time_series([(x_nodes, x_adj), (y_nodes, y_adj), (z_nodes, z_adj)])
        self.assertClose(0., actual_loss)

        # consider a more complicated example in the dynamical system, where
        # our loss should still be zero
        x_nodes = ['a', 'a', 'a']
        x_adj   = [[1, 2], [], []]
        y_nodes = ['a', 'a', 'b', 'b', 'a', 'b', 'b']
        y_adj   = [[1, 4], [2, 3], [], [], [5, 6], [], []]
        actual_loss = model.loss_over_time_series([(x_nodes, x_adj), (y_nodes, y_adj)])
        self.assertClose(0., actual_loss)


    def test_predict_macrostep(self):
        # we configure a network precisely such that it should produce a simple
        # dynamical system over trees, where each 'a' is replaced with the
        # subtree 'a(b, b)' and each 'b' is replaced with a.
        model = DynSysTEN()

        # test the case of a single b
        nodes = ['b']
        adj   = [[]]
        edits, nodes, adj = model.predict_macrostep(nodes, adj)
        expected_nodes = ['a']
        expected_adj   = [[]]
        self.assertEqual(nodes, expected_nodes)
        self.assertEqual(adj, expected_adj)
        self.assertEqual(edits, [[te.Replacement(0, 'a')]])

        # test the case of a single a
        nodes = ['a']
        adj   = [[]]
        edits, nodes, adj = model.predict_macrostep(nodes, adj)
        expected_nodes = ['a', 'b', 'b']
        expected_adj   = [[1, 2], [], []]
        self.assertEqual(nodes, expected_nodes)
        self.assertEqual(adj, expected_adj)
        self.assertEqual(edits, [[], [te.Insertion(0, 0, 'b')], [te.Insertion(0, 1, 'b')]])

        # test a more complicated tree
        nodes = ['b', 'a', 'b']
        adj   = [[1], [2], []]
        edits, nodes, adj = model.predict_macrostep(nodes, adj)

        expected_nodes = ['a', 'a', 'a', 'b']
        expected_adj   = [[1], [2, 3], [], []]
        self.assertEqual(nodes, expected_nodes)
        self.assertEqual(adj, expected_adj)
        self.assertEqual(edits, [[te.Replacement(0, 'a'), te.Replacement(2, 'a')], [te.Insertion(1, 1, 'b')]])

        # and another one
        nodes = ['a', 'b', 'a']
        adj   = [[1], [2], []]
        edits, nodes, adj = model.predict_macrostep(nodes, adj)

        expected_nodes = ['a', 'a', 'a', 'b', 'b', 'b']
        expected_adj   = [[1, 5], [2], [3, 4], [], [], []]
        self.assertEqual(nodes, expected_nodes)
        self.assertEqual(adj, expected_adj)
        self.assertEqual(edits, [[te.Replacement(1, 'a')], [te.Insertion(0, 1, 'b'), te.Insertion(2, 0, 'b')], [te.Insertion(2, 1, 'b')]])

        # and a final one
        nodes = ['a', 'a', 'a']
        adj   = [[1, 2], [], []]
        edits, nodes, adj = model.predict_macrostep(nodes, adj)

        expected_nodes = ['a', 'a', 'b', 'b', 'a', 'b', 'b']
        expected_adj   = [[1, 4], [2, 3], [], [], [5, 6], [], []]
        self.assertEqual(nodes, expected_nodes)
        self.assertEqual(adj, expected_adj)
        self.assertEqual(edits, [[], [te.Insertion(1, 0, 'b'), te.Insertion(3, 0, 'b')], [te.Insertion(1, 1, 'b'), te.Insertion(4, 1, 'b')]])


    def test_learning(self):
        # test a simple tree dynamical system where every 'a' that does not yet have two
        # children  is replaced with the motif 'a(b, b)' and where every b is replaced
        # by an a
        alphabet = ['a', 'b']

        # set up a tree edit network
        model = ten.TEN(1, alphabet, dim_hid = 32, dim_in_extra = 3, nonlin = torch.nn.Tanh(), dim_memory = 2)
        # use the DynSysTEN as a teacher network
        teacher = DynSysTEN()
        def sample_time_series(T = 2, init_sym = None):
            # sample an initial tree
            if init_sym is None:
                if random.random() > 0.5:
                    init_sym = 'b'
                else:
                    init_sym = 'a'
            nodes = [init_sym]
            adj = [[]]
            #  and let the DynSysTEN run on it for T time steps
            trees = [(nodes, adj)]
            for t in range(T):
                _, nodes, adj = teacher.predict_macrostep(nodes, adj)
                trees.append((nodes, adj))
            return trees

        # set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1E-2, weight_decay=1E-5)
        # learn
        loss_avg = None
        max_epochs = 10000
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            # sample time series
            trees = sample_time_series()
            # compute loss
            loss  = model.loss_over_time_series(trees)
            # do backprop
            loss.backward()
            # do optimization step
            optimizer.step()
            # compute moving average of loss
            if loss_avg is None:
                loss_avg = loss.item()
            else:
                loss_avg = 0.9 * loss_avg + 0.1 * loss.item()
            if loss_avg < 1E-3:
                break
            if (epoch + 1) % 100 == 0:
                print('loss avg after %d epochs: %g' % (epoch+1, loss_avg))

        # verify correctness
        for sym in alphabet:
            trees = sample_time_series(init_sym = sym)
            for t in range(len(trees)-1):
                nodes, adj = trees[t]
                expected_nodes, expected_adj = trees[t+1]
                _, actual_nodes, actual_adj = model.predict_macrostep(nodes, adj)
                self.assertEqual(expected_nodes, actual_nodes)
                self.assertEqual(expected_adj, actual_adj)

    def test_to_edits(self):
        # set up an example tree
        nodes = ['a', 'b', 'c', 'd', 'e']
        adj   = [[1, 4], [2, 3], [], [], []]
        max_degree = 2

        alphabet = nodes
        # test first simple cases of single replacements, deletions, and insertions

        # single deletion
        delta = torch.zeros(len(nodes))
        delta[1] = -1.
        types   = torch.eye(len(nodes))
        Cidx = torch.zeros(len(nodes), max_degree + 1)
        Cnum = torch.zeros(len(nodes), max_degree)

        expected_edits = te.Script([te.Deletion(1)])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet)
        self.assertEqual(expected_edits, actual_edits)

        # single replacement
        delta[1]    = 0.25
        types[1][0] = 2.

        expected_edits = te.Script([te.Replacement(1, alphabet[0])])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet)
        self.assertEqual(expected_edits, actual_edits)

        # single insertion without children at various indices
        delta[1]   = +1.
        Cidx[1, 0] = 1.

        expected_edits = te.Script([te.Insertion(1, 0, alphabet[0])])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet)
        self.assertEqual(expected_edits, actual_edits)

        Cidx[1, 0] = 0.
        Cidx[1, 1] = 1.

        expected_edits = te.Script([te.Insertion(1, 1, alphabet[0])])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet)
        self.assertEqual(expected_edits, actual_edits)

        Cidx[1, 1] = 0.
        Cidx[1, 2] = 1.

        expected_edits = te.Script([te.Insertion(1, 2, alphabet[0])])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet)
        self.assertEqual(expected_edits, actual_edits)

        # do a deletion followed by an insertion, such that the insertion index
        # needs to be adjusted
        delta[0] = -1.
        expected_edits = te.Script([te.Deletion(0), te.Insertion(0, 2, alphabet[0])])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet)
        self.assertEqual(expected_edits, actual_edits)
        # the adjustment should not take place, if we explicitly shut it off
        expected_edits = te.Script([te.Deletion(0), te.Insertion(1, 2, alphabet[0])])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet, adjust_indices = False)
        self.assertEqual(expected_edits, actual_edits)

        # delete a child before the inserted node, such that the child index
        # needs to be adjusted
        delta[0] = 0.
        delta[2] = -1.
        expected_edits = te.Script([te.Deletion(2), te.Insertion(1, 1, alphabet[0])])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet)
        self.assertEqual(expected_edits, actual_edits)
        # the adjustment should not take place, if we explicitly shut it off
        expected_edits = te.Script([te.Deletion(2), te.Insertion(1, 2, alphabet[0])])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet, adjust_indices = False)
        self.assertEqual(expected_edits, actual_edits)

        # do another insertion after the first insertion
        delta[3] = +1.
        expected_edits = te.Script([te.Deletion(2), te.Insertion(1, 1, alphabet[0]), te.Insertion(2, 0, alphabet[3])])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet)
        self.assertEqual(expected_edits, actual_edits)
        # the adjustment should not take place, if we explicitly shut it off
        expected_edits = te.Script([te.Deletion(2), te.Insertion(1, 2, alphabet[0]), te.Insertion(3, 0, alphabet[3])])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet, adjust_indices = False)
        self.assertEqual(expected_edits, actual_edits)

        # do an insertion which would use a child as a grand child, but that child
        # is deleted before
        delta = torch.zeros(len(nodes))
        types = torch.eye(len(nodes))
        Cidx  = torch.zeros(len(nodes), max_degree + 1)
        Cnum  = torch.zeros(len(nodes), max_degree)
        delta[1] = +1.
        Cidx[1, 0] = 1.
        Cnum[1, :] = 1.
        delta[2] = -1.

        expected_edits = te.Script([te.Deletion(2), te.Insertion(1, 0, alphabet[1], 1)])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet)
        self.assertEqual(expected_edits, actual_edits)
        # the adjustment should not take place, if we explicitly shut it off
        expected_edits = te.Script([te.Deletion(2), te.Insertion(1, 0, alphabet[1], 2)])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet, adjust_indices = False)
        self.assertEqual(expected_edits, actual_edits)

        # now do a somewhat more complicated case with multiple deletions and
        # insertions
        delta = torch.zeros(len(nodes))
        types = torch.eye(len(nodes))
        Cidx  = torch.zeros(len(nodes), max_degree + 1)
        Cnum  = torch.zeros(len(nodes), max_degree)
        # we insert a new c child of a which takes e as child
        delta[0] = +1.
        types[0, 2] = 1.
        Cidx[0, 1] = 1.
        Cnum[0, :] = 1.
        # we delete b, which also means that the child index of our
        # insertion before needs to take the previous children of b into
        # account
        delta[1] = -1.
        # we replace c with b
        types[2][1] = 2.
        # we insert a new e child of d
        delta[3] = +1.
        types[3][4] = 2.

        expected_edits = te.Script([te.Replacement(2, alphabet[1]), te.Deletion(1), te.Insertion(0, 2, alphabet[2], 1), te.Insertion(2, 0, alphabet[4], 0)])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet)
        self.assertEqual(expected_edits, actual_edits)

        # validate the result tree as well
        expected_nodes = ['a', 'b', 'd', 'e', 'c', 'e']
        expected_adj   = [[1, 2, 4], [], [3], [], [5], []]
        actual_nodes, actual_adj = actual_edits.apply(nodes, adj)
        self.assertEqual(expected_nodes, actual_nodes)
        self.assertEqual(expected_adj, actual_adj)

        # the adjustment should not take place, if we explicitly shut it off
        expected_edits = te.Script([te.Replacement(2, alphabet[1]), te.Deletion(1), te.Insertion(0, 1, alphabet[2], 1), te.Insertion(3, 0, alphabet[4], 0)])
        actual_edits   = ten.to_edits(nodes, adj, delta, types, Cidx, Cnum, alphabet, adjust_indices = False)
        self.assertEqual(expected_edits, actual_edits)

if __name__ == '__main__':
    unittest.main()
