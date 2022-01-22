#!/usr/bin/python3
"""
Tests the kernel time series prediction classes.

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
import kernel_time_series_prediction as ktsp


__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.1.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

class KernelTreePredictorTest(unittest.TestCase):

    def test_constant(self):
        # as a test case, we consider the mock example from the paper,
        # where we have two string time series as training data, namely
        # ['a', 'aac'] and ['b', 'bbc']
        time_series = [
            [(['a'], [[]]), (['a', 'a', 'c'], [[1], [2], []])],
            [(['b'], [[]]), (['b', 'b', 'c'], [[1], [2], []])]
        ]
        T = np.sum(list(map(len, time_series)))
        # train the model
        model = ktsp.KernelTreePredictor(psi = 0.5)
        model.fit(time_series)
        # first, test trivial prediction cases, where we request predictions
        # for exact copies of our trianing data. In these cases, we expect
        # the result to be exactly the training output
        i = 0
        for s in range(len(time_series)):
            seq = time_series[s]
            for t in range(len(seq)):
                alpha, nodes, adj = model.predict(*seq[t])
                expected_alpha = np.zeros(T)
                expected_alpha[i] = 1.
                np.testing.assert_almost_equal(expected_alpha, alpha, 3)
                if t < len(seq)-1:
                    expected_nodes, expected_adj = seq[t+1]
                else:
                    expected_nodes, expected_adj = seq[t]
                self.assertEqual(expected_nodes, nodes)
                self.assertEqual(expected_adj, adj)
                i += 1

        # next, test the input from the paper, namely 'ab'. In that case,
        # both inputs are equally far away and the compromise edit is to
        # insert a 'c' in the end. Whether we actually apply that edit,
        # though, depends on the radial basis function bandwidth
        model.psi = 1.
        alpha, nodes, adj = model.predict(['a', 'b'], [[1], []])
        expected_d = np.array([1., 2., 1., 2.])
        expected_k = np.exp(-0.5 * expected_d ** 2 / model.psi ** 2)
        expected_alpha = np.dot(model._Kinv, expected_k)
        np.testing.assert_almost_equal(expected_alpha, alpha, 3)
        expected_nodes = ['a', 'b', 'c']
        expected_adj   = [[1], [2], []]
        self.assertEqual(expected_nodes, nodes)
        self.assertEqual(expected_adj, adj)

if __name__ == '__main__':
    unittest.main()
