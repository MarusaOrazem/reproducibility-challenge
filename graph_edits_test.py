#!/usr/bin/python3
"""
Tests graph edits.

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
import numpy as np
import graph_edits

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

class TestGraphEdits(unittest.TestCase):

    def test_mapping_to_edits(self):
        # test an isomorphic case where no edit should occur
        X = np.array([[0.], [1.], [2.]])
        A = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        Y = np.array([[0.], [2.], [1.]])
        B = np.array([[0., 0., 1.], [0., 0., 1.], [1., 1., 0.]])
        phi = np.array([0, 2, 1])

        script = graph_edits.mapping_to_edits(A, X, B, Y, phi)
        self.assertEqual([], script)

        # test a case with a replacement
        Y[1] = 1.

        script = graph_edits.mapping_to_edits(A, X, B, Y, phi)
        self.assertEqual([graph_edits.NodeReplacement(2, np.array([1.]))], script)

        # test a case with a node insertion
        Y = np.array([[0.], [2.], [3.], [1.]])
        B = np.array([[0., 0., 0., 1.], [0., 0., 1., 1.], [0., 1., 0., 0.], [1., 1., 0., 0.]])
        phi = np.array([0, 3, 1])

        script = graph_edits.mapping_to_edits(A, X, B, Y, phi, directed = False)
        self.assertEqual([graph_edits.NodeInsertion(2, np.array([3.]), directed = False)], script)

        # test a case with a node deletion
        Y = np.array([[2.], [1.]])
        B = np.array([[0., 1.], [1., 0.]])
        phi = np.array([-1, 1, 0])

        script = graph_edits.mapping_to_edits(A, X, B, Y, phi)
        self.assertEqual([graph_edits.NodeDeletion(0)], script)

        # test a case with edge edits
        Y = np.array([[0.], [2.], [1.]])
        B = np.array([[0., 1., 0.], [0., 0., 1.], [1., 1., 0.]])
        phi = np.array([0, 2, 1])

        script = graph_edits.mapping_to_edits(A, X, B, Y, phi)
        self.assertEqual([graph_edits.EdgeInsertion(0, 2), graph_edits.EdgeDeletion(0, 1)], script)

        # test a case with undirected edge edits
        B = np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        script = graph_edits.mapping_to_edits(A, X, B, Y, phi)
        self.assertEqual([graph_edits.EdgeInsertion(0, 2), graph_edits.EdgeInsertion(2, 0)], script)
        script = graph_edits.mapping_to_edits(A, X, B, Y, phi, directed = False)
        self.assertEqual([graph_edits.EdgeInsertion(0, 2, directed = False)], script)

    def test_mapping_to_edits_random_graphs(self):
        # first, consider random graphs and permute nodes around, which should
        # yield no edits
        n = 10
        p = 0.5
        for i in range(100):
            # sample a random graph
            B = np.random.rand(n, n)
            B = 0.5 * (B + B.T)
            B[B >  1. - p] = 1.
            B[B <= 1. - p] = 0.
            Y = np.zeros((n, 1))
            # permute it randomly
            phi = np.random.permutation(n)
            A = B[phi, :][:, phi]
            X = Y[phi, :]

            # check the edits
            script = graph_edits.mapping_to_edits(A, X, B, Y, phi)
            self.assertEqual([], script)

        # then compare two random graphs and check that the edit script
        # actually transforms the input into the output graph

        for i in range(100):
            # sample two random graphs
            A = np.random.rand(n, n)
            A = 0.5 * (A + A.T)
            A -= np.diag(np.diag(A))
            A[A >  1. - p] = 1.
            A[A <= 1. - p] = 0.
            X = np.random.randint(2, size = (n, 1))

            m = np.random.randint(n-2, n+2)
            B = np.random.rand(m, m)
            B = 0.5 * (B + B.T)
            B -= np.diag(np.diag(B))
            B[B >  1. - p] = 1.
            B[B <= 1. - p] = 0.
            Y = np.random.randint(2, size = (m, 1))

            # use a random permutation
            if n <= m:
                phi = np.random.permutation(m)[:n]
            else:
                phi = np.full(n, -1)
                phi[:m] = np.random.permutation(m)

            # obtain edits
            script, phi = graph_edits.mapping_to_edits(A, X, B, Y, phi, directed = False, return_full_phi = True)

            # apply the script
            Bpred, Ypred = graph_edits.apply_script(script, A, X)

            if not np.array_equal(B[phi, :][:, phi], Bpred):

                print('input')
                print(np.concatenate((A, X), 1))
                print('expected output')
                print(np.concatenate((B[phi, :][:, phi], Y[phi, :]), 1))
                print('script')
                print(script)

                print('actual output')
                print(np.concatenate((Bpred, Ypred), 1))
                print('diff')
                print(np.concatenate((B[phi, :][:, phi] - Bpred, Y[phi, :] - Ypred), 1))
                raise ValueError('stop')

            # check equality
            np.testing.assert_array_equal(B[phi, :][:, phi], Bpred)
            np.testing.assert_array_equal(Y[phi], Ypred)


if __name__ == '__main__':
    unittest.main()
