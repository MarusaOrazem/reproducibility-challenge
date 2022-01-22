#!/usr/bin/python3
"""
Tests the degree rules dataset.

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
import random
import numpy as np
import graph_edits as ge
import degree_rules as dr

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestDegreeRules(unittest.TestCase):

    def numpy_list_assert_equal(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for t in range(len(actual)):
            self.assertTrue(np.sum(np.abs(expected[t] - actual[t])) < 1E-3, "Expected %s but got %s in time step %d" % (expected[t], actual[t], t))

    def test_next_step(self):
        A = np.array([[0, 1], [1, 0]])
        edits, delta, Epsilon = dr.next_step(A, 3)
        self.assertEqual([ge.NodeInsertion(0, np.array([0., 0., 1.]), False)], edits)
        np.testing.assert_array_equal(np.array([1., 1.]), delta)
        np.testing.assert_array_equal(np.array([[0., 0.], [0., 0.]]), Epsilon)

        A = np.array([[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]])
        edits, delta, Epsilon = dr.next_step(A)
        self.assertEqual([ge.EdgeInsertion(0, 2, False)], edits)
        np.testing.assert_array_equal(np.ones(len(A)), delta)
        Epsilon_expected = np.dot(A, A)
        Epsilon_expected[Epsilon_expected > 0.5] = 1.
        Epsilon_expected[A > 0.5] = 0.
        np.fill_diagonal(Epsilon_expected, 0)
        np.testing.assert_array_equal(Epsilon_expected, Epsilon)

        A = np.array([[0, 1, 1, 0, 1], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]])
        edits, delta, Epsilon = dr.next_step(A)
        self.assertEqual([ge.EdgeInsertion(1, 3, False)], edits)
        np.testing.assert_array_equal(np.array([0., 1., 0., 1., 1.]), delta)
        Epsilon_expected = np.dot(A, A)
        Epsilon_expected[Epsilon_expected > 0.5] = 1.
        Epsilon_expected[A > 0.5] = 0.
        np.fill_diagonal(Epsilon_expected, 0)
        np.testing.assert_array_equal(Epsilon_expected, Epsilon)

        A = np.array([[0, 1, 1, 0, 1], [1, 0, 1, 1, 0], [1, 1, 0, 1, 0], [0, 1, 1, 0, 1], [1, 0, 0, 1, 0]])
        edits, delta, Epsilon = dr.next_step(A)
        self.assertEqual([ge.EdgeInsertion(1, 4, False)], edits)
        np.testing.assert_array_equal(np.array([0., 0., 0., 0., 1.]), delta)
        Epsilon_expected = np.dot(A, A)
        Epsilon_expected[Epsilon_expected > 0.5] = 1.
        Epsilon_expected[A > 0.5] = 0.
        np.fill_diagonal(Epsilon_expected, 0)
        np.testing.assert_array_equal(Epsilon_expected, Epsilon)

    def test_growth(self):
        # test the case of growing a 4-clique from an initial seed node
        A_init = np.array([[0.]])
        As, Xs, deltas, Epsilons = dr.generate_time_series(A_init)
        # check result
        # adjacency matrices
        As_expected = [
            A_init,
            np.array([[0., 1.], [1., 0.]]),
            np.array([[0., 1., 1.], [1., 0., 0.], [1., 0., 0.]]),
            np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]),
            np.array([[0., 1., 1., 1.], [1., 0., 1., 0.], [1., 1., 0., 0.], [1., 0., 0., 0.]]),
            np.array([[0., 1., 1., 1.], [1., 0., 1., 1.], [1., 1., 0., 0.], [1., 1., 0., 0.]]),
            np.array([[0., 1., 1., 1.], [1., 0., 1., 1.], [1., 1., 0., 1.], [1., 1., 1., 0.]])
        ]
        self.numpy_list_assert_equal(As_expected, As)

        # node feature matrices
        Xs_expected = [np.eye(1, 4), np.eye(2, 4), np.eye(3, 4), np.eye(3, 4), np.eye(4), np.eye(4), np.eye(4)]
        self.numpy_list_assert_equal(Xs_expected, Xs)

        # node edit vectors
        deltas_expected = [np.ones(1), np.ones(2), np.ones(3), np.ones(3), np.array([0., 1., 1., 1.]), np.array([0., 0., 1., 1.]), np.zeros(4)]
        self.numpy_list_assert_equal(deltas_expected, deltas)

        # edge edit matrices
        Epsilons_expected = [
            np.zeros(1),
            np.zeros(2),
            np.array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.]]),
            np.zeros(3),
            np.array([[0., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 0., 1.], [0., 1., 1., 0.]]),
            np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 1., 0.]]),
            np.zeros(4)
        ]
        self.numpy_list_assert_equal(Epsilons_expected, Epsilons)

    def test_shrink(self):
        # test a case where we have a graph with a central node that first
        # shrinks and then grows again
        A_init = np.array([
            [0., 1., 0., 0., 1.],
            [1., 0., 1., 0., 1.],
            [0., 1., 0., 1., 1.],
            [0., 0., 1., 0., 1.],
            [1., 1., 1., 1., 0.]
        ])
        As, Xs, deltas, Epsilons = dr.generate_time_series(A_init)
        # check result
        # adjacency matrices
        As_expected = [
            A_init,
            np.array([[0., 1., 0., 0.], [1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.]]),
            np.array([[0., 1., 1., 0.], [1., 0., 1., 0.], [1., 1., 0., 1.], [0., 0., 1., 0.]]),
            np.array([[0., 1., 1., 1.], [1., 0., 1., 0.], [1., 1., 0., 1.], [1., 0., 1., 0.]]),
            np.array([[0., 1., 1., 1.], [1., 0., 1., 1.], [1., 1., 0., 1.], [1., 1., 1., 0.]])
        ]
        self.numpy_list_assert_equal(As_expected, As)
        # node feature matrices
        Xs_expected = [np.eye(5, 20), np.eye(4, 20), np.eye(4, 20), np.eye(4, 20), np.eye(4, 20)]
        self.numpy_list_assert_equal(Xs_expected, Xs)
        # node edit vectors
        deltas_expected = [
            np.array([1., 0., 0., 1., -1.]),
            np.ones(4),
            np.array([1., 1., 0., 1.]),
            np.array([0., 1., 0., 1.]),
            np.zeros(4)
        ]
        self.numpy_list_assert_equal(deltas_expected, deltas)
        # edge edit matrices
        Epsilons_expected = [
            np.array([[0., 0., 1., 1., 0.], [0., 0., 0., 1., 0.], [1., 0., 0., 0., 0.], [1., 1., 0., 0., 0.], [0., 0., 0., 0., 0.]]),
            np.array([[0., 0., 1., 0.], [0., 0., 0., 1.], [1., 0., 0., 0.], [0., 1., 0., 0.]]),
            np.array([[0., 0., 0., 1.], [0., 0., 0., 1.], [0., 0., 0., 0.], [1., 1., 0., 0.]]),
            np.array([[0., 0., 0., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.], [0., 1., 0., 0.]]),
            np.zeros((4, 4))
        ]
        self.numpy_list_assert_equal(Epsilons_expected, Epsilons)

    def test_large_scale(self):
        # check that all graphs converge to disconnected 4-cliques
        N = 16
        four = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
        for trial in range(100):
            As, _, _, _ = dr.generate_time_series_from_random_matrix(N)
            # consider only the final adjacency matrix
            A = As[-1]
            # search it via depth first search
            remaining = set(range(len(A)))
            while remaining:
                # aggregate the connected component of i
                i = min(remaining)
                C = []
                stk = [i]
                while(stk):
                    i = stk.pop()
                    if(i not in remaining):
                        continue
                    remaining.remove(i)
                    C.append(i)
                    for j in np.where(A[i, :])[0]:
                        if(j in remaining):
                            stk.append(j)
                # ensure that the sub-adjacency matrix for this connected
                # component is indeed a 4-clique
                A_C = A[C, :][:, C]
                self.assertTrue(np.sum(np.abs(four - A_C)) < 1E-3)

if __name__ == '__main__':
    unittest.main()
