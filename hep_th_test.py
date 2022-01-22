#!/usr/bin/python3
"""
Tests the HEP-Th dataset.

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
import numpy as np
import pytorch_graph_edit_networks as gen
import hep_th

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

class TestHEP_Th(unittest.TestCase):

    def test_read_graph_from_csv(self):
        A, idxs = hep_th.read_graph_from_csv('hep-th/graphs/1992_1.csv')
        self.assertEqual(len(A), len(idxs))
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                self.assertEqual(A[i, j], A[j, i])

    def test_add_graph(self):
        A = np.ones((3, 3), dtype=int)
        I = np.array([1, 2, 4], dtype=int)
        B = np.ones((3, 3), dtype=int)
        J = np.array([2, 3, 5], dtype=int)
        C_expected = np.array([
            [0, 1, 0, 1, 0],
            [1, 0, 1, 1, 1],
            [0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0]
        ], dtype=int)
        K_expected = np.array([1, 2, 3, 4, 5], dtype=int)

        C, K = hep_th.add_graph(A, I, B, J)
        np.testing.assert_array_equal(K, K_expected)
        np.testing.assert_array_equal(C, C_expected)

        C, K = hep_th.add_graph(B, J, A, I)
        np.testing.assert_array_equal(K, K_expected)
        np.testing.assert_array_equal(C, C_expected)

    def test_teaching_protocol(self):
        # set up input
        A = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0]
            ])
        I = [1, 2, 3, 4]
        B = np.array([
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0]
            ])
        J = [1, 3, 4, 5]

        # get predicted teaching protocol
        delta, Epsilon = hep_th._teaching_protocol(A, I, B, J)

        # get expected teaching protocol
        delta_true = np.array([0, -1, 0, +1])
        Epsilon_true = np.array([
            [0, 0, -1, 0],
            [0, 0, 0, 0],
            [-1, 0, 0, +1],
            [0, 0, +1, 0]
            ])

        # compare
        np.testing.assert_array_equal(delta, delta_true)
        np.testing.assert_array_equal(Epsilon, Epsilon_true)

    def test_teaching_protocol2(self):
        A, I, delta, Epsilon = hep_th.teaching_protocol(2000, 2, 12, 12)
        self.assertEqual(A.shape[0], A.shape[1])
        self.assertEqual(A.shape[0], len(I))
        self.assertEqual(A.shape[0], len(delta))
        self.assertEqual(A.shape, Epsilon.shape)

    def test_compute_loss(self):
        model = gen.GEN(2, 1, 16, filter_edge_edits = True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1E-3, weight_decay=1E-3)

        for year in range(1993, 1993+1):
            for month in range(1, 12+1):
                optimizer.zero_grad()
                loss = hep_th.compute_loss(model, year, month)
                self.assertTrue(loss > 10.)

#    def test_compute_max_size(self):
#        max_size = 0
#        for month in hep_th._month_strings:
#            A, I = hep_th.read_graph_from_csv('hep-th/graphs/%s.csv' % month)
#            if len(I) > max_size:
#                max_size = len(I)
#        print('The maximum graph size for a single month in the HEP-Th dataset is %d' % max_size)

#    def test_author_overlap(self):
#        # test non-overlapping time windows of 12 month and check how many
#        # authors leave and enter

#        # build the node sets for each year
#        author_sets = []
#        for year in range(1992, 2003):
#            authors = set()
#            for month in range(1, 12):
#                _, I = hep_th.read_graph_from_csv('hep-th/graphs/%d_%d.csv' % (year, month))
#                authors.update(I)
#            author_sets.append(authors)

#        # compute fraction of leaving and entering authors, as well as the
#        # Jaccard similarity
#        for i in range(len(author_sets)-1):
#            leaving  = len(author_sets[i].difference(author_sets[i+1])) / len(author_sets[i])
#            entering = len(author_sets[i+1].difference(author_sets[i])) / len(author_sets[i+1])
#            jacc = len(author_sets[i].intersection(author_sets[i+1])) / len(author_sets[i].union(author_sets[i+1]))
#            print('Year %d: leaving: %g%%, entering: %g%%, jaccard: %g' % (1992 + i, leaving * 100, entering * 100, jacc))

    def test_compute_map(self):
        # we take the example of a graph with three nodes, where we predict the
        # edges (0, 1), (0, 2), (1, 0), and (1, 2), with scores 3, 1, 2, and 1
        # respectively, and where the true edges are (0, 1) and (1, 2).
        predicted_adj = [[(1, 3.), (2, 1.)], [(0, 1.), (2, 2.)], []]
        actual_adj    = [[1], [2], []]
        # accordingly, our mean average precision should be 1., because the
        # highest-ranking predictions are always correct
        res = hep_th.computeMAP(predicted_adj, actual_adj)
        self.assertTrue(abs(res - 1.) < 1E-3)
        res = hep_th.computeMAP(predicted_adj, actual_adj, 1)
        self.assertTrue(abs(res - 1.) < 1E-3)

        # conversely, if we switch the scores for one prediction, the MAP
        # should drop to 0.75, because we get one prediction out of four
        # wrong, so to speak.
        predicted_adj = [[(1, 3.), (2, 1.)], [(0, 2.), (2, 1.)], []]
        res = hep_th.computeMAP(predicted_adj, actual_adj)
        self.assertTrue(abs(res - .75) < 1E-3)
        # if we restrict to the top-ranking prediction, the MAP should drop to
        # 0.5
        res = hep_th.computeMAP(predicted_adj, actual_adj, 1)
        self.assertTrue(abs(res - .5) < 1E-3)

    def test_evaluate_model(self):
        model = gen.GEN(2, 1, 16, filter_edge_edits = True)
        model.eval()

        for year in range(1993, 1993+1):
            for month in range(1, 12+1):
                res = hep_th.evaluate_model(model, year, month)
                self.assertTrue(res < 0.7)

if __name__ == '__main__':
    unittest.main()
