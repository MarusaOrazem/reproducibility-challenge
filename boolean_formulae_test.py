#!/usr/bin/python3
"""
Tests the Boolean formulae dataset.

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
import torch
from edist.alignment import Alignment
import edist.tree_utils as tu
import pytorch_tree_edit_networks as ten
import boolean_formulae

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

class MockTEN(torch.nn.Module):

    def __init__(self, delta, types, Cidx, Cnum):
        super(MockTEN, self).__init__()
        self.delta = delta
        self.types = types
        self.Cidx  = Cidx
        self.Cnum  = Cnum
        self._dim_in_extra = 3
        self._dim_memory = 0

    def forward(self, nodes, adj, X):
        return self.delta, self.types, self.Cidx, self.Cnum

class TestBoolean(unittest.TestCase):

    def test_generate_tree(self):
        # sample a few trees and ensure that they all have at most the
        # maximum number of binary operators
        max_ops = 3
        for r in range(100):
            nodes, adj = boolean_formulae._generate_tree(max_ops)
            self.assertTrue(nodes.count('and') + nodes.count('or') <= max_ops)
            self.assertTrue(len(nodes) > 1)
            self.assertTrue(len(nodes) <= 8)

    def test_simplify(self):
        # test each rule with a few elementary examples

        # rule 1: and(x, False) is equivalent to False.
        nodes = ['and', 'x', 'False']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['False'], [[]]), time_series[1])

        nodes = ['and', 'False', 'x']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['False'], [[]]), time_series[1])

        # rule 2: and(x, True) is equivalent to x.
        nodes = ['and', 'x', 'True']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['x'], [[]]), time_series[1])

        nodes = ['and', 'True', 'x']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['x'], [[]]), time_series[1])

        # rule 3: or(x, True) is equivalent to True.
        nodes = ['or', 'x', 'True']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['True'], [[]]), time_series[1])

        nodes = ['or', 'True', 'x']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['True'], [[]]), time_series[1])

        # rule 4: or(x, False) is equivalent to x.
        nodes = ['or', 'x', 'False']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['x'], [[]]), time_series[1])

        nodes = ['or', 'False', 'x']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['x'], [[]]), time_series[1])

        # rule 5: and(x, x) is equivalent to x.
        nodes = ['and', 'x', 'x']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['x'], [[]]), time_series[1])

        # rule 6: or(x, x) is equivalent to x.
        nodes = ['or', 'x', 'x']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['x'], [[]]), time_series[1])

        # rule 7: and(x, not_x) is equivalent to False.
        nodes = ['and', 'x', 'not_x']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['False'], [[]]), time_series[1])

        nodes = ['and', 'not_x', 'x']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['False'], [[]]), time_series[1])

        # rule 8: or(x, not(x)) is equivalent to True.
        nodes = ['or', 'x', 'not_x']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['True'], [[]]), time_series[1])

        nodes = ['or', 'not_x', 'x']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['True'], [[]]), time_series[1])

        # now test cases where rules apply to less trivial trees
        # rule 1
        nodes = ['and', 'or', 'x', 'y', 'False']
        adj   = [[1, 4], [2, 3], [], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(3, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['and', 'x', 'y', 'False'], [[1, 2, 3], [], [], []]), time_series[1])
        self.assertEqual((['False'], [[]]), time_series[2])

        # rule 3
        nodes = ['or', 'and', 'x', 'y', 'False']
        adj   = [[1, 4], [2, 3], [], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['and', 'x', 'y'], [[1, 2], [], []]), time_series[1])

        # rule 4
        nodes = ['or', 'not_x', 'True']
        adj   = [[1, 2], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['True'], [[]]), time_series[1])

        # rule 5
        nodes = ['or', 'x', 'x', 'x', 'x']
        adj   = [[1, 2, 3, 4], [], [], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['x'], [[]]), time_series[1])

        # rule 7
        nodes = ['and', 'y', 'y', 'x', 'not_x']
        adj   = [[1, 2, 3, 4], [], [], [], []]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(2, len(time_series))
        self.assertEqual((nodes, adj), time_series[0])
        self.assertEqual((['False'], [[]]), time_series[1])

        # finally, perform tests where the resulting time series should have
        # multiple steps
        nodes = ['and', 'y', 'or', 'x', 'not_x']
        adj   = [[1, 2], [], [3, 4], [], []]
        expected_time_series = [
            (nodes, adj),
            (['and', 'y', 'True'], [[1, 2], [], []]),
            (['y'], [[]])
        ]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(expected_time_series, time_series)

        nodes = ['or', 'y', 'and', 'x', 'not_x']
        adj   = [[1, 2], [], [3, 4], [], []]
        expected_time_series = [
            (nodes, adj),
            (['or', 'y', 'False'], [[1, 2], [], []]),
            (['y'], [[]])
        ]
        time_series = boolean_formulae._simplify(nodes, adj)
        self.assertEqual(expected_time_series, time_series)

    def test_generate_time_series(self):
        # just generate a few time series and ensure that no error occurs
        # and that at least a few randomly sampled trees can be simplified
        non_trivial = 0
        max_depth = 0
        for r in range(100):
            time_series = boolean_formulae.generate_time_series()
            self.assertTrue(len(time_series) > 0)
            if len(time_series) > 1:
                non_trivial += 1
            else:
                continue
            # make sure that all simplification steps still yield trees
            for tree in time_series:
                adj = tree[1]
                r = tu.root(adj)
                depth = tree_depth(adj, r)
                if depth > max_depth:
                    max_depth = depth
        self.assertTrue(non_trivial > 3)
        self.assertTrue(max_depth > 3)

    def test_boolean_alignment(self):
        # test a few non-trivial cases

        # finally, perform tests where the resulting time series should have
        # multiple steps
        nodes = ['and', 'y', 'or', 'x', 'not_x']
        adj   = [[1, 2], [], [3, 4], [], []]
        expected_alignment = Alignment()
        expected_alignment.append_tuple(0, 0)
        expected_alignment.append_tuple(1, 1)
        expected_alignment.append_tuple(2, 2)
        expected_alignment.append_tuple(3, -1)
        expected_alignment.append_tuple(4, -1)
        alignment = boolean_formulae.boolean_alignment(nodes, adj, None, None)
        self.assertEqual(expected_alignment, alignment)

        nodes = ['and', 'False', 'or', 'x', 'y']
        adj   = [[1, 2], [], [3, 4], [], []]
        expected_alignment = Alignment()
        expected_alignment.append_tuple(0, 0)
        expected_alignment.append_tuple(1, 1)
        expected_alignment.append_tuple(2, -1)
        expected_alignment.append_tuple(3, 2)
        expected_alignment.append_tuple(4, 3)
        alignment = boolean_formulae.boolean_alignment(nodes, adj, None, None)
        self.assertEqual(expected_alignment, alignment)

        nodes = ['and', 'False', 'x', 'y', 'False']
        adj   = [[1, 2, 3, 4], [], [], [], []]
        expected_alignment = Alignment()
        expected_alignment.append_tuple(0, -1)
        expected_alignment.append_tuple(1, 0)
        expected_alignment.append_tuple(2, -1)
        expected_alignment.append_tuple(3, -1)
        expected_alignment.append_tuple(4, -1)
        alignment = boolean_formulae.boolean_alignment(nodes, adj, None, None)
        self.assertEqual(expected_alignment, alignment)

    def test_compute_loss(self):

        # rule 1: and(x, False) is equivalent to False.
        nodes = ['and', 'x', 'False']
        adj   = [[1, 2], [], []]

        # set up a mockup model which provides the correct output
        types = torch.zeros(3, len(boolean_formulae.alphabet))
        types[2][boolean_formulae.alphabet.index('False')] = 10.
        mock_model = MockTEN(
            torch.tensor([-1., -1., 0.]),
            types,
            None,
            None)
        # which should lead to a loss of zero
        loss = boolean_formulae.compute_loss(mock_model, [(nodes, adj)])
        self.assertTrue(loss.item() < 1E-2)

        # but a different mock model should lead to a different loss
        types = torch.zeros(3, len(boolean_formulae.alphabet))
        types[0][boolean_formulae.alphabet.index('False')] = 10.
        types[2][boolean_formulae.alphabet.index('True')] = 10.
        mock_model = MockTEN(
            torch.tensor([0., -1., 0.]),
            types,
            None,
            None)
        loss = boolean_formulae.compute_loss(mock_model, [(nodes, adj)])
        self.assertTrue(abs(11. - loss.item()) < 1E-2)

        # rule 8: or(x, not(x)) is equivalent to True.
        nodes = ['or', 'x', 'not_x']
        adj   = [[1, 2], [], []]

        # set up a mockup model which provides the correct output
        types = torch.zeros(3, len(boolean_formulae.alphabet))
        types[0][boolean_formulae.alphabet.index('True')] = 10.
        mock_model = MockTEN(
            torch.tensor([0., -1., -1.]),
            types,
            None,
            None)

        # which should lead to a loss of zero
        loss = boolean_formulae.compute_loss(mock_model, [(nodes, adj)])
        self.assertTrue(loss.item() < 1E-2)


        # but a different mock model should lead to a different loss
        mock_model = MockTEN(
            torch.tensor([-1., -1., -1.]),
            types,
            None,
            None)
        loss = boolean_formulae.compute_loss(mock_model, [(nodes, adj)])
        self.assertTrue(abs(0.75 * 0.75 - loss.item()) < 1E-2)


    def test_learning(self):
        net = ten.TEN(num_layers = 2, alphabet = boolean_formulae.alphabet,
                  dim_hid = 64, dim_in_extra = 6 + 1, nonlin = torch.nn.ReLU(),
                  skip_connections = False, dim_memory = 0)
        optimizer = torch.optim.Adam(net.parameters(), lr=1E-3, weight_decay=1E-8)

        # start training
        loss_avg = None
        learning_curve = []
        epochs = 0
        while loss_avg is None or loss_avg > 1E-3:
            optimizer.zero_grad()
            # sample a time series
            time_series = boolean_formulae.generate_time_series(2)
            if len(time_series) < 2:
                continue
            # compute the time series loss
            loss_obj = boolean_formulae.compute_loss(net, time_series)
            # compute the gradient
            loss_obj.backward()
            # perform an optimizer step
            optimizer.step()
            # compute a new moving average over the loss
            if loss_avg is None:
                loss_avg = loss_obj.item()
            else:
                loss_avg = loss_avg * 0.9 + 0.1 * loss_obj.item()
            if((epochs+1) % 100 == 0):
                print('loss avg after %d epochs: %g' % (epochs+1, loss_avg))

            epochs += 1

        # verify that some prototypical cases work after training
        trees = [
            (['root', 'and', 'x', 'False'], [[1], [2, 3], [], []]),
            (['root', 'and', 'x', 'True'], [[1], [2, 3], [], []]),
            (['root', 'or', 'x', 'True'], [[1], [2, 3], [], []]),
            (['root', 'or', 'x', 'False'], [[1], [2, 3], [], []]),
            (['root', 'and', 'x', 'x'], [[1], [2, 3], [], []]),
            (['root', 'or', 'x', 'x'], [[1], [2, 3], [], []]),
            (['root', 'and', 'x', 'not_x'], [[1], [2, 3], [], []]),
            (['root', 'or', 'x', 'not_x'], [[1], [2, 3], [], []]),
        ]
        expected_trees = [
            (['root', 'False'], [[1], []]),
            (['root', 'x'], [[1], []]),
            (['root', 'True'], [[1], []]),
            (['root', 'x'], [[1], []]),
            (['root', 'x'], [[1], []]),
            (['root', 'x'], [[1], []]),
            (['root', 'False'], [[1], []]),
            (['root', 'True'], [[1], []])
        ]

        for i in range(len(trees)):
            nodes, adj = trees[i]
            nodes_exp, adj_exp = expected_trees[i]
            _, nodes_act, adj_act = boolean_formulae.predict_step(net, nodes, adj)
            self.assertEqual(nodes_exp, nodes_act)
            self.assertEqual(adj_exp, adj_act)


def tree_depth(adj, i = 0):
    max_depth = 0
    for j in adj[i]:
        depth = tree_depth(adj, j)
        if depth > max_depth:
            max_depth = depth
    return max_depth + 1

if __name__ == '__main__':
    unittest.main()
