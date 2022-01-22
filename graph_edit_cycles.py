"""
Generates data for the graph edit cycles data set where we consider
graphlets of size 4 and evolve them over time.

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

import numpy as np
import graph_edits as ge

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

# The initial states for the graph cycles as tuples, where the
# adjacency matrix is the first and the attribute matrix is the second entry.
# The node attribute matrix just consists of the node index
cycles_init = [
    (np.array([[0]]), np.array([[1, 0, 0, 0]])),
    (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])),
    (np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
]

# The graph edits constructing the remainder of the cycles. Each cycle
# is defined by one list of edits
cycles_edits = [
     # The first cycle just adds a node and deletes it again
    [ge.NodeInsertion(0, np.array([0, 1, 0, 0]), False), ge.NodeDeletion(1)],
    # The second cycle inserts a node, connects it to the start
    # node, and then deletes it again
    [ge.NodeInsertion(2, np.array([0, 0, 0, 1]), False), ge.EdgeInsertion(0, 3, False), ge.NodeDeletion(3)],
    # The third cycle inserts a node, fully connects the graph, and then
    # deletes the node again
    [ge.NodeInsertion(2, np.array([0, 0, 0, 1]), False), ge.EdgeInsertion(0, 3, False), ge.EdgeInsertion(1, 3, False), ge.NodeDeletion(3)]
]

# to avoid having to apply every edit all the time again, we cache a
# complete version of every graph cycle
cycles_As = []
cycles_Xs = []
cycles_deltas = []
for c in range(len(cycles_init)):
    As = [cycles_init[c][0]]
    Xs = [cycles_init[c][1]]
    deltas = []
    for e in range(len(cycles_edits[c])-1):
        deltas.append(cycles_edits[c][e].score(len(As[-1])))
        A, X = cycles_edits[c][e].apply(As[-1], Xs[-1])
        As.append(A)
        Xs.append(X)
    deltas.append(cycles_edits[c][-1].score(len(As[-1])))
    cycles_As.append(As)
    cycles_Xs.append(Xs)
    cycles_deltas.append(deltas)

def generate_time_series(c, t_0, T):
    """ Runs the graph edit cycle c for T time steps, starting from step t.

    Parameters
    ----------
    c: int
        The index of the cycle (0, 1, or 2).
    t_0: int
        The starting time step within the cycle.
    T: int
        The length of the output time series.

    Returns
    -------
    As: list
        A time series of adjacency matrices.
    Xs: list
        A time series of node attribute matrices.
    deltas: list
        A time series of targets, i.e. tuples of node scores and
        edge scores, where -1 indicates a deletion and +1 indicates
        an insertion.

    """
    As = []
    Xs = []
    deltas = []
    T_c = len(cycles_As[c])
    for t in range(t_0, T + t_0):
        As.append(cycles_As[c][t % T_c])
        Xs.append(cycles_Xs[c][t % T_c])
        deltas.append(cycles_deltas[c][t % T_c])
    return As, Xs, deltas

