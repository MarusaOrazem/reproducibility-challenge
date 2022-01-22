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
import edist.tree_edits as te
import edist.tree_utils as tu
import pytorch_tree_edit_networks as ten
import peano_addition

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
            nodes, adj = peano_addition._generate_tree(max_ops)
            self.assertTrue(nodes.count('+') <= max_ops)
            self.assertTrue(len(nodes) >= 2)
            self.assertTrue(len(nodes) <= 8)

    def test_simplify(self):
        # first, test trees which should not be simplified
        nodes = ['root', '0']
        adj   = [[1], []]
        expected_time_series = [(nodes, adj)]
        time_series = peano_addition._simplify(nodes, adj)
        self.assertEqual(expected_time_series, time_series)

        nodes = ['root', '1']
        adj   = [[1], []]
        expected_time_series = [(nodes, adj)]
        time_series = peano_addition._simplify(nodes, adj)
        self.assertEqual(expected_time_series, time_series)

        nodes = ['root', '3']
        adj   = [[1], []]
        expected_time_series = [(nodes, adj)]
        time_series = peano_addition._simplify(nodes, adj)
        self.assertEqual(expected_time_series, time_series)

        # then, test a simple resolve of a succ chain
        nodes = ['root', 'succ', 'succ', '1']
        adj   = [[1], [2], [3], []]
        expected_time_series = [(nodes, adj),
            (['root', 'succ', '2'], [[1], [2], []]),
            (['root', '3'], [[1], []])
            ]
        time_series = peano_addition._simplify(nodes, adj)
        self.assertEqual(expected_time_series, time_series)

        # then, test a complete addition
        nodes = ['root', '+', '3', '2']
        adj   = [[1], [2, 3], [], []]
        expected_time_series = [(nodes, adj),
            (['root', '+', '3', 'succ', '1'], [[1], [2, 3], [], [4], []]),
            (['root', '+', 'succ', '3', '1'], [[1], [2, 4], [3], [], []]),
            (['root', '+', '4', 'succ', '0'], [[1], [2, 3], [], [4], []]),
            (['root', '+', 'succ', '4', '0'], [[1], [2, 4], [3], [], []]),
            (['root', '5'], [[1], []]),
            ]
        time_series = peano_addition._simplify(nodes, adj)
        self.assertEqual(expected_time_series, time_series)

        # perform a test with two nested additions
        nodes = ['root', '+', '+', '5', 'succ', '0', '0']
        adj   = [[1], [2, 6], [3, 4], [], [5], [], []]
        expected_time_series = [(nodes, adj),
            (['root', '+', 'succ', '5', '0'], [[1], [2, 4], [3], [], []]),
            (['root', '6'], [[1], []])
            ]
        time_series = peano_addition._simplify(nodes, adj)
        self.assertEqual(expected_time_series, time_series)

        # finally test a tree with multiple additions
        nodes = ['root', '+', '+', '7', '2', '+', '1', '1']
        adj   = [[1], [2, 5], [3, 4], [], [], [6, 7], [], []]
        expected_time_series = [(nodes, adj),
            (['root', '+', '+', '7', 'succ', '1', '+', '1', 'succ', '0'], [[1], [2, 6], [3, 4], [], [5], [], [7, 8], [], [9], []]),
            (['root', '+', '+', 'succ', '7', '1', '+', 'succ', '1', '0'], [[1], [2, 6], [3, 5], [4], [], [], [7, 9], [8], [], []]),
            (['root', '+', '+', '8', 'succ', '0', '2'], [[1], [2, 6], [3, 4], [], [5], [], []]),
            (['root', '+', '+', 'succ', '8', '0', 'succ', '1'], [[1], [2, 6], [3, 5], [4], [], [], [7], []]),
            (['root', '+', 'succ', '9', '1'], [[1], [2, 4], [3], [], []]),
            (['root', '+', '0', 'succ', '0'], [[1], [2, 3], [], [4], []]),
            (['root', '+', 'succ', '0', '0'], [[1], [2, 4], [3], [], []]),
            (['root', '1'], [[1], []]),
            ]
        time_series = peano_addition._simplify(nodes, adj)
        self.assertEqual(expected_time_series, time_series)

    def test_peano_alignment(self):
        nodes = ['root', '1']
        adj   = [[1], []]
        expected_alignment = Alignment()
        expected_alignment.append_tuple(0, 0)
        expected_alignment.append_tuple(1, 1)
        alignment = peano_addition.peano_alignment(nodes, adj, None, None)
        self.assertEqual(expected_alignment, alignment)

        nodes = ['root', 'succ', '1']
        adj   = [[1], [2], []]
        expected_alignment = Alignment()
        expected_alignment.append_tuple(0, 0)
        expected_alignment.append_tuple(1, -1)
        expected_alignment.append_tuple(2, 1)
        alignment = peano_addition.peano_alignment(nodes, adj, None, None)
        self.assertEqual(expected_alignment, alignment)

        nodes = ['root', '+', '2', '1']
        adj   = [[1], [2, 3], [], []]
        expected_alignment = Alignment()
        expected_alignment.append_tuple(0, 0)
        expected_alignment.append_tuple(1, 1)
        expected_alignment.append_tuple(2, 2)
        expected_alignment.append_tuple(-1, 3)
        expected_alignment.append_tuple(3, 4)
        alignment = peano_addition.peano_alignment(nodes, adj, None, None)
        self.assertEqual(expected_alignment, alignment)

        nodes = ['root', '+', '2', 'succ', '0']
        adj   = [[1], [2, 3], [], [4], []]
        expected_alignment = Alignment()
        expected_alignment.append_tuple(0, 0)
        expected_alignment.append_tuple(1, 1)
        expected_alignment.append_tuple(-1, 2)
        expected_alignment.append_tuple(2, 3)
        expected_alignment.append_tuple(3, -1)
        expected_alignment.append_tuple(4, 4)
        alignment = peano_addition.peano_alignment(nodes, adj, None, None)
        self.assertEqual(expected_alignment, alignment)

        nodes = ['root', '+', '2', '0']
        adj   = [[1], [2, 3], [], []]
        expected_alignment = Alignment()
        expected_alignment.append_tuple(0, 0)
        expected_alignment.append_tuple(1, -1)
        expected_alignment.append_tuple(2, 1)
        expected_alignment.append_tuple(3, -1)
        alignment = peano_addition.peano_alignment(nodes, adj, None, None)
        self.assertEqual(expected_alignment, alignment)

        nodes = ['root', '+', '1', 'succ', '+', '0', '0']
        adj   = [[1], [2, 3], [], [4], [5, 6], [], []]
        expected_alignment = Alignment()
        expected_alignment.append_tuple(0, 0)
        expected_alignment.append_tuple(1, 1)
        expected_alignment.append_tuple(-1, 2)
        expected_alignment.append_tuple(2, 3)
        expected_alignment.append_tuple(3, -1)
        expected_alignment.append_tuple(4, -1)
        expected_alignment.append_tuple(5, 4)
        expected_alignment.append_tuple(6, -1)
        alignment = peano_addition.peano_alignment(nodes, adj, None, None)
        self.assertEqual(expected_alignment, alignment)

        nodes = ['root', '+', '+', '1', 'succ', '0', '0']
        adj   = [[1], [2, 6], [3, 4], [], [5], [], []]
        expected_alignment = Alignment()
        expected_alignment.append_tuple(0, 0)
        expected_alignment.append_tuple(1, -1)
        expected_alignment.append_tuple(2, 1)
        expected_alignment.append_tuple(-1, 2)
        expected_alignment.append_tuple(3, 3)
        expected_alignment.append_tuple(4, -1)
        expected_alignment.append_tuple(5, 4)
        expected_alignment.append_tuple(6, -1)
        alignment = peano_addition.peano_alignment(nodes, adj, None, None)
        self.assertEqual(expected_alignment, alignment)

    def test_compute_loss(self):
        # set up a mockup model which has a constant output
        mock_model = MockTEN(
            torch.tensor([-1., 0., -1.]),
            torch.tensor([[0., 0., 0.], [0., 0., 10.], [0., 0., 0.]]),
            None,
            None)
        # verify zero loss on a simple tree
        nodes = ['+', '1', '0']
        adj   = [[1, 2], [], []]

        loss = peano_addition.compute_loss(mock_model, [(nodes, adj)])
        self.assertTrue(loss.item() < 1E-2)

        # if we test a different model, it should have a nonzero loss
        mock_model = MockTEN(
            torch.tensor([0., 0., 0.]),
            torch.tensor([[0., 0., 0.], [0., 0., 10.], [0., 0., 0.]]),
            None,
            None)

        loss = peano_addition.compute_loss(mock_model, [(nodes, adj)])
        self.assertTrue(abs(loss.item() - 2.) < 1E-2)

        # test resolving a succ
        types = torch.zeros(3, len(peano_addition.alphabet))
        types[0, peano_addition.alphabet.index('root')] = 10.
        types[2, peano_addition.alphabet.index('3')] = 10.
        mock_model = MockTEN(
            torch.tensor([0., -1., 0.]),
            types,
            None,
            None)

        nodes = ['root', 'succ', '2']
        adj   = [[1], [2], []]

        loss = peano_addition.compute_loss(mock_model, [(nodes, adj)])
        self.assertTrue(loss.item() < 1E-2)

        # if we test a different model, it should have a nonzero loss
        types = torch.zeros(3, len(peano_addition.alphabet))
        types[0, peano_addition.alphabet.index('root')] = 10.
        types[2, peano_addition.alphabet.index('2')] = 10.
        mock_model = MockTEN(
            torch.tensor([0., -1., 0.]),
            types,
            None,
            None)

        loss = peano_addition.compute_loss(mock_model, [(nodes, adj)])
        self.assertTrue(abs(loss.item() - 10.) < 1E-2)

    def test_predict_step(self):
        # set up a mockup model which has a constant output
        mock_model = MockTEN(
            torch.tensor([0., -1., 0.]),
            torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 1.]]),
            None,
            None)
        # verify the resulting prediction on a simple tree
        nodes = ['a', 'b', 'c']
        adj   = [[1], [2], []]

        script_act, nodes_act, adj_act = peano_addition.predict_step(mock_model, nodes, adj, ['a', 'b', 'c'])

        script_exp = te.Script()
        script_exp.append(te.Deletion(1))
        nodes_exp, adj_exp = script_exp.apply(nodes, adj)

        self.assertEqual(script_exp, script_act)
        self.assertEqual(nodes_exp, nodes_act)
        self.assertEqual(adj_exp, adj_act)

        # set up a mockup model which has a non-trivial constant output
        mock_model = MockTEN(
            torch.tensor([+1., 0., -1.]),
            torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 0., 1.]]),
            torch.tensor([[1., 0., 0., 0.]]),
            torch.tensor([[1., 0., 0.]]))
        # verify the resulting prediction on a simple tree
        nodes = ['a', 'b', 'c']
        adj   = [[1], [2], []]

        script_act, nodes_act, adj_act = peano_addition.predict_step(mock_model, nodes, adj, ['a', 'b', 'c'])

        script_exp = te.Script()
        script_exp.append(te.Replacement(1, 'c'))
        script_exp.append(te.Insertion(0, 0, 'a', 1))
        script_exp.append(te.Deletion(3))
        nodes_exp, adj_exp = script_exp.apply(nodes, adj)

        self.assertEqual(script_exp, script_act)
        self.assertEqual(nodes_exp, nodes_act)
        self.assertEqual(adj_exp, adj_act)

    def test_learning(self):
        net = ten.TEN(num_layers = 2, alphabet = peano_addition.alphabet,
                  dim_hid = 64, dim_in_extra = 2 + 1, nonlin = torch.nn.ReLU(),
                  skip_connections = False)
        optimizer = torch.optim.Adam(net.parameters(), lr=1E-3, weight_decay=1E-8)

        # start training
        loss_avg = None
        learning_curve = []
        epochs = 0
        while loss_avg is None or loss_avg > 1E-2:
            optimizer.zero_grad()
            # sample a time series
            time_series = peano_addition.generate_time_series(2, 1)
            if len(time_series) < 2:
                continue
            # compute the time series loss
            loss_obj = peano_addition.compute_loss(net, time_series)
            # loss_obj = net.loss_over_time_series(time_series, custom_alignment = peano_addition.peano_alignment)
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

        # verify that new time series are correctly predicted
        for r in range(10):
            # sample a time series
            time_series = peano_addition.generate_time_series(2, 1)
            for t in range(len(time_series)-1):
                # let the network predict
                nodes, adj = time_series[t]
                _, nodes_act, adj_act = peano_addition.predict_step(net, nodes, adj)
                # compare with expected tree
                nodes_exp, adj_exp = time_series[t+1]
                self.assertEqual(nodes_exp, nodes_act)
                self.assertEqual(adj_exp, adj_act)

        # verify that three prototypical cases work after training
        trees = [
            (['root', '+', 'succ', '1', '0'], [[1], [2, 4], [3], [], []]),
            (['root', '+', '1', '1'], [[1], [2, 3], [], []]),
            (['root', '+', '1', 'succ', '0'], [[1], [2, 3], [], [4], []])
        ]
        expected_trees = [
            (['root', '2'], [[1], []]),
            (['root', '+', '1', 'succ', '0'], [[1], [2, 3], [], [4], []]),
            (['root', '+', 'succ', '1', '0'], [[1], [2, 4], [3], [], []])
        ]

        for i in range(len(trees)):
            nodes, adj = trees[i]
            nodes_exp, adj_exp = expected_trees[i]
            _, nodes_act, adj_act = peano_addition.predict_step(net, nodes, adj)
            self.assertEqual(nodes_exp, nodes_act)
            self.assertEqual(adj_exp, adj_act)


if __name__ == '__main__':
    unittest.main()
