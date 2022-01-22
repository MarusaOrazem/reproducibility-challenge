#!/usr/bin/python3
"""
Tests the graph edit network implementation.

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

import unittest
import time
import random
import torch
import numpy as np
import graph_edits as ge
import pytorch_graph_edit_networks as gen

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

_SKIP_LONG_TESTS = True

class TestGENs(unittest.TestCase):

    def test_to_edits(self):
        # zero inputs should yield an empy script
        A = torch.zeros(3, 3)
        delta = torch.zeros(3)
        Epsilon = torch.zeros(3, 3)
        script = gen.to_edits(A, delta, Epsilon)
        self.assertEqual(0, len(script))
        # if we delete edges that are not present, this should remain
        Epsilon[0, 1] = -1.
        script = gen.to_edits(A, delta, Epsilon)
        self.assertEqual(0, len(script))
        # if that edge was present, we should get an edge deletion
        A[0, 1] = 1.
        script = gen.to_edits(A, delta, Epsilon)
        expected_script = [ge.EdgeDeletion(0, 1)]
        self.assertEqual(expected_script, script)
        # A positive edge edit score should yield edge insertions
        Epsilon[1, 2] = 1.
        script = gen.to_edits(A, delta, Epsilon)
        expected_script.append(ge.EdgeInsertion(1, 2))
        self.assertEqual(expected_script, script)
        # If we delete a node to which we have a positive edge edit score,
        # the edge should not be inserted
        delta[2] = -1.
        script = gen.to_edits(A, delta, Epsilon)
        expected_script[1] = ge.NodeDeletion(2)
        self.assertEqual(expected_script, script)
        # check node insertions as well
        delta[1] = +1.
        script = gen.to_edits(A, delta, Epsilon)
        expected_script.insert(1, ge.NodeInsertion(1, 0.))
        self.assertEqual(expected_script, script)
        # ensure that node insertions occur in ascending order
        delta[0] = +1.
        script = gen.to_edits(A, delta, Epsilon)
        expected_script.insert(1, ge.NodeInsertion(0, 0.))
        self.assertEqual(expected_script, script)
        # and that node deletions occur in descending order
        delta[1] = -1.
        script = gen.to_edits(A, delta, Epsilon)
        expected_script = [ge.NodeInsertion(0, 0.), ge.NodeDeletion(2), ge.NodeDeletion(1)]
        self.assertEqual(expected_script, script)

    def test_gen_loss_crossent(self):
        loss_fun = gen.GEN_loss_crossent()
        # test node loss
        expected_losses = [
            #-1         0   1 (v vs. v_true)
            [0.0, 0.5, 1.5], # -1
            [0.5, 0.0, 0.5], # 0
            [1.5, 0.5, 0.0]  # +1
        ]
        vals = [-1., 0.,+1.]

        for i in range(len(vals)):
            for j in range(len(vals)):
                v = torch.tensor([vals[j]])
                Epsilon = torch.tensor([[0.]])
                v_true = torch.tensor([vals[i]])
                Epsilon_true = torch.tensor([[0.]])
                A            = torch.tensor([[0.]])
                expected_loss = expected_losses[i][j]
                actual_loss  = loss_fun(v, Epsilon, v_true, Epsilon_true, A).item()
                self.assertTrue(abs(expected_loss - actual_loss) < 1E-2, "expected node loss %g for predicted value %g versus actual value %g but got %g" % (expected_loss, vals[j], vals[i], actual_loss))

        # test edge loss
        expected_losses = [
            # edge is not yet present
            [
                #-1  0   1 (Epsilon vs. Epsilon_true)
                [0., 0., 1.], # 0
                [3., 1., 0.]  # +1
            ],
            # edge is present
            [
                #-1  0   1 (Epsilon vs. Epsilon_true)
                [0., 1., 3.], # -1
                [1., 0., 0.], # 0
            ],
        ]
        true_vals = [[0., 1.], [-1., 0.]]

        for e in [0, 1]:
            for i in range(len(true_vals[e])):
                for j in range(len(vals)):
                    v = torch.tensor([0., 0.])
                    Epsilon = torch.tensor([[0., vals[j]], [vals[j], 0.]])
                    v_true = torch.tensor([0., 0.])
                    Epsilon_true = torch.tensor([[0., true_vals[e][i]], [true_vals[e][i], 0.]])
                    A            = torch.tensor([[0, e], [e, 0]], dtype=torch.float)
                    expected_loss = expected_losses[e][i][j]
                    actual_loss  = loss_fun(v, Epsilon, v_true, Epsilon_true, A).item()
                    self.assertTrue(abs(expected_loss - actual_loss) < 1E-2, "expected edge loss %g for predicted value %g versus actual value %g but got %g" % (expected_loss, vals[j], true_vals[e][i], actual_loss))

        # test filter loss
        expected_losses = [
            # -1 + 1 (edge_filters vs Epsilon_true)
            [0., 4.], # -1
            [4., 0.]
        ]
        vals = [-1., +1.]

        A = torch.tensor([[0, 0], [0, 0]], dtype=torch.float)

        for i in range(len(true_vals[e])):
            for j in range(len(vals)):
                v = torch.tensor([0., 0.])
                v_true = torch.tensor([0., 0.])
                if vals[i] > 0:
                    Epsilon = torch.tensor([[vals[j]]])
                else:
                    Epsilon = torch.tensor([[]])
                if vals[j] > 0:
                    Epsilon_true = torch.tensor([[0., 1.], [0., 0.]])
                else:
                    Epsilon_true = torch.tensor([[0., 0.], [0., 0.]])
                edge_filters = torch.tensor([[vals[i], -1.], [-1., vals[i]]])
                expected_loss = expected_losses[i][j]
                actual_loss  = loss_fun(v, Epsilon, v_true, Epsilon_true, A, edge_filters).item()
                self.assertTrue(abs(expected_loss - actual_loss) < 1E-2, "expected filter loss %g for predicted value %g versus actual value %g but got %g" % (expected_loss, vals[j], vals[i], actual_loss))

    def test_gen_loss_perceptron(self):
        loss_fun = gen.GEN_loss()
        # test node loss
        expected_losses = [
            #-1         0   1 (delta_pred vs. delta_true)
            [0.,        1., 4.], # -1
            [0.75 ** 2, 0., 0.75 ** 2], # 0
            [4.,        1., 0.]  # +1
        ]
        vals = [-1., 0.,+1.]

        for i in range(len(vals)):
            for j in range(len(vals)):
                delta_pred = torch.tensor([vals[j]])
                Epsilon_pred = torch.tensor([[0.]])
                delta_true = torch.tensor([vals[i]])
                Epsilon_true = torch.tensor([[0.]])
                A            = torch.tensor([[0.]])
                expected_loss = expected_losses[i][j]
                actual_loss  = loss_fun(delta_pred, Epsilon_pred, delta_true, Epsilon_true, A).item()
                self.assertTrue(abs(expected_loss - actual_loss) < 1E-3, "expected node loss %g for predicted value %g versus actual value %g but got %g" % (expected_loss, vals[j], vals[i], actual_loss))

        # test edge loss
        expected_losses = [
            # edge is not yet present
            [
                #-1  0   1 (Epsilon_pred vs. Epsilon_true)
                [0., 0., 1.], # 0
                [4., 1., 0.]  # +1
            ],
            # edge is present
            [
                #-1  0   1 (Epsilon_pred vs. Epsilon_true)
                [0., 1., 4.], # -1
                [1., 0., 0.], # 0
            ],
        ]
        true_vals = [[0., 1.], [-1., 0.]]

        for e in [0, 1]:
            for i in range(len(true_vals[e])):
                for j in range(len(vals)):
                    delta_pred = torch.tensor([0., 0.])
                    Epsilon_pred = torch.tensor([[0., vals[j]], [vals[j], 0.]])
                    delta_true = torch.tensor([0., 0.])
                    Epsilon_true = torch.tensor([[0., true_vals[e][i]], [true_vals[e][i], 0.]])
                    A            = torch.tensor([[0, e], [e, 0]], dtype=torch.float)
                    expected_loss = 2 * expected_losses[e][i][j]
                    actual_loss  = loss_fun(delta_pred, Epsilon_pred, delta_true, Epsilon_true, A).item()
                    self.assertTrue(abs(expected_loss - actual_loss) < 1E-3, "expected edge loss %g for predicted value %g versus actual value %g but got %g" % (expected_loss, vals[j], true_vals[e][i], actual_loss))

        # test a case where multiple nodes and edges should be edited
        delta_pred = torch.tensor([1.25,-2.])
        Epsilon_pred = torch.tensor([[0., -2.], [0., 0.]])
        delta_true = torch.tensor([0., +1.])
        Epsilon_true = torch.tensor([[0., 0.], [1., 0.]])
        A            = torch.tensor([[0., 1.], [0., 0.]])
        expected_loss = 1. + 9. + 4. + 1.
        actual_loss  = loss_fun(delta_pred, Epsilon_pred, delta_true, Epsilon_true, A).item()
        self.assertTrue(abs(expected_loss - actual_loss) < 1E-3, "expected loss %g but got %g" % (expected_loss, actual_loss))

        # test a few cases with edge filtering
        delta_pred = torch.tensor([0., 0.])
        Epsilon_pred = torch.tensor([[-1.]])
        delta_true = torch.tensor([0., 0.])
        Epsilon_true = torch.tensor([[0., -1.], [+0., 0.]])
        A            = torch.tensor([[0., 1.], [0., 0.]])
        edge_filters = torch.tensor([[1.,-1.], [-1., 1.]])
        expected_loss = 0.
        actual_loss  = loss_fun(delta_pred, Epsilon_pred, delta_true, Epsilon_true, A, edge_filters).item()
        self.assertTrue(abs(expected_loss - actual_loss) < 1E-3, "expected loss %g but got %g" % (expected_loss, actual_loss))

        Epsilon_pred = torch.tensor([[0., -1.]])
        edge_filters = torch.tensor([[1., .5], [-1., 1.]])
        expected_loss = 1. * len(A)
        actual_loss  = loss_fun(delta_pred, Epsilon_pred, delta_true, Epsilon_true, A, edge_filters).item()
        self.assertTrue(abs(expected_loss - actual_loss) < 1E-3, "expected loss %g but got %g" % (expected_loss, actual_loss))


    def test_degree_computation(self):
        if _SKIP_LONG_TESTS:
            return
        print('degree computation test')
        # test whether our GCN implementation is able to compute the in degree
        # of every node in an undirected graph

        # one GCN layer should suffice here, because we can simply add up
        # the input, provided that the node representation is '1'.
        net = gen.GCN(1, 1)

        # instantiate optimizer and loss function
        loss_fun = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1E-3, weight_decay=1E-3)

        # learn
        loss_threshold = 1E-3
        epochs = 1
        loss_avg = 100.
        loss_avgs = []
        while loss_avg > loss_threshold:
            optimizer.zero_grad()

            # generate a new adjacency matrix of random size
            N = random.randrange(4, 16)
            # generate random edge entries
            A = torch.rand(N, N)
            # round to the nearest integer
            A = torch.round(A)
            # initialize node embedding
            X_init = torch.ones(N, 1)

            # compute the current net output
            X = net(X_init, A)
            # compute the loss
            degrees = torch.sum(A, 0).unsqueeze(1)
            loss = loss_fun(degrees, X)
            # compute the gradient
            loss.backward()
            # perform an optimizer step
            optimizer.step()
            # compute a new moving average over the loss
            loss_avg = loss_avg * 0.9 + 0.1 * loss.item()
            loss_avgs.append(loss_avg)
            if(epochs % 100 == 0):
                print('loss avg after %d epochs: %g' % (epochs, loss_avg))
            epochs += 1

    def test_neighbor_computation(self):
        if _SKIP_LONG_TESTS:
            return
        print('shared neighbors computation test')
        # test whether our GEN implementation is able to compute the number
        # of common neighbors two nodes share

        # we should only need one layer for this task, which computes the
        # set of neighboring nodes. The edge edit score computation then
        # produces an inner product of this set representation, which implements
        # a set intersection
        n = 16
        net = gen.GEN(1, n, n, nonlin = torch.nn.Sigmoid())

        # instantiate optimizer and loss function
        loss_fun = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1E-3, weight_decay=1E-5)

        # learn
        loss_threshold = 1E-2
        max_epochs = 50000
        epochs = 1
        loss_avg = 100.
        loss_avgs = []
        while loss_avg > loss_threshold and epochs < max_epochs:
            optimizer.zero_grad()
            # generate a new random graph of random size
            N = random.randrange(4, 16)
            A = torch.rand(N, N)
            A = 0.5 * (A + A.t())
            A = torch.round(A)
            # initialize node embedding
            X_init = torch.eye(N, n)

            # compute the current net output
            _, E = net(A, X_init)

            # compute the ground truth
            E_expected  = torch.mm(A, A)
            E_expected -= torch.diag(torch.diag(E_expected))

            # compute the loss
            loss = loss_fun(E_expected.detach(), E)
            # compute the gradient
            loss.backward()
            # perform an optimizer step
            optimizer.step()
            # compute a new moving average over the loss
            loss_avg = loss_avg * 0.9 + 0.1 * loss.item()
            loss_avgs.append(loss_avg)
            if(epochs % 100 == 0):
                print('loss avg after %d epochs: %g' % (epochs, loss_avg))
            epochs += 1
        self.assertTrue(epochs < max_epochs)

    def test_degree_rules(self):
        # test whether our GEN implementation is able to solve the degree rules
        # task if we set the parameters manually
        n = 8
        net = gen.GEN(num_layers = 1, dim_in = n, dim_hid = n+2, nonlin = torch.nn.ReLU())
        # We can solve the task if the first n neurons represent the
        # neighbors by summing up their indices in one-hot coding and the
        # last two neurons represent whether a node's degree is larger than 3
        # (rule 1) or smaller than 3 (rule 3).
        f = 100.
        # put everything to zero first
        net._layers[0]._U.weight[:, :] = 0.
        net._layers[0]._V.weight[:, :] = 0.
        net._layers[0]._W.weight[:, :] = 0.
        net._layers[0]._U.bias[:] = 0.
        net._layers[0]._V.bias[:] = 0.
        net._layers[0]._W.bias[:] = 0.
        net._edge_u.weight[:] = 0.
        net._edge_u.bias[:] = 0.
        net._edge_v.weight[:] = 0.
        net._edge_v.bias[:] = 0.
        net._edge_w.weight[:] = 0.
        net._edge_w.bias[:] = 0.
        net._node.weight[:, :] = 0.
        net._node.bias[:] = 0.
        # rule 1 representation: Node deletion if degree larger 3
        # can be done by doing -sigmoid(sum_incoming 1 - 3.5)
        net._layers[0]._U.weight[n, :] = 1. / f
        net._layers[0]._U.bias[n] = -3.5 / f
        net._node.weight[0,n] = -2. * f
        # rule 2 representation: Edge insertion if two nodes share the
        # same neighbor.
        # Can be done by first representing the set of neighbors as
        # sigmoid(sum_incoming index) and then inferring the edges as
        # the inner product of those vectors
        net._layers[0]._U.weight[:n, :] = torch.eye(n)
        net._edge_w.weight[:] = 1.
        # rule 3 representation: Node insertion if degree smaller 3.
        # can be done by doing +sigmoid(-sum_incoming 1 + 2.5)
        net._layers[0]._U.weight[n+1, :] = -1. / f
        net._layers[0]._U.bias[n+1] = +2.5 / f
        net._node.weight[0,n+1] = 2. * f

        # test the network
        A = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]])
        degrees = np.sum(A, 1)
        delta = np.zeros(len(A))
        Epsilon = np.zeros(A.shape)
        degrees = np.sum(A, axis=1)
        delta[degrees > 3] = -1. # rule 1
        Epsilon[np.logical_and(np.dot(A, A) > 0.5, np.logical_not(A))] = 1. # rule 2
        np.fill_diagonal(Epsilon, 0.) # correct for self-connections
        delta[degrees < 3] = 1. # rule 3
        # compare to network output
        delta_pred, Epsilon_pred = net(torch.tensor(A, dtype=torch.float), torch.eye(len(A), n))

        # compute loss
        loss_fun = gen.GEN_loss()
        loss = loss_fun(delta_pred, Epsilon_pred, torch.tensor(delta), torch.tensor(Epsilon), torch.tensor(A))
        self.assertTrue(loss.item() < 1E-3, 'unexpectedly high loss: %g. expected delta = %s but got %s; Epsilon = %s but got %s' % (loss.item(), delta, delta_pred.detach().numpy(), Epsilon, Epsilon_pred.detach().numpy()))


    def test_game_of_life(self):
        if _SKIP_LONG_TESTS:
            return
        print('game of life test')
        # check whether we can implement the basic rules of game of life
        # if the grid connectivity and the current node state on each
        # grid point is given. In particular, the node state x can be zero
        # (dead) or one (alive) and our network output should be the
        # change in the node state y, i.e. 0 if nothing changes, +1 if a
        # dead node gets alive and -1 if an alive node dies.
        # We frame this as a classification problem where every output
        # value >= 0.5 is clamped to one, every output <= -0.5 is clamped to
        # -1 and every output in the interval (-0.5, 0.5) is clamped to zero.
        # Further, if an alive node gets a positive signal, nothing changes
        # and if a dead node receives a negative signal, nothing changes,
        # such that our overall loss is:
        #
        # loss = ReLU((1 - 2*x - 2*y_true) * y_pred + abs(y_true))
        #
        # which is positive only if a dead node that should stay dead receies
        # a positive signal, or if a dead node that should get alive receives
        # a signal smaller than +1, or if an alive node that should stay
        # alive receives a negative signal, or if an alive node that should die
        # receives a signal larger than -1.

        # define the loss function
        def loss_fun(X, delta_pred, delta_true):
            sign = torch.sign(1. - 2*X - 2*delta_true)
            offset = torch.abs(delta_true)
            return torch.sum(torch.nn.functional.relu(sign * delta_pred + offset))

        # hyper-parameters for game of life
        grid_size = 10
        m = grid_size * grid_size
        p = 0.3
        max_steps = 8
        # generate adjacency matrix
        A = torch.zeros((m, m))
        for i in range(grid_size):
            for j in range(grid_size):
                k = i * grid_size + j
                if i < grid_size - 1:
                    l = (i+1) * grid_size + j
                    A[k, l] = 1.
                    A[l, k] = 1.
                    if j < grid_size - 1:
                        l = (i+1) * grid_size + j + 1
                        A[k, l] = 1.
                        A[l, k] = 1.
                if j < grid_size - 1:
                    l = i * grid_size + j + 1
                    A[k, l] = 1.
                    A[l, k] = 1.
                    if i > 0:
                        l = (i-1) * grid_size + j + 1
                        A[k, l] = 1.
                        A[l, k] = 1.

        # first, we manually set up a GEN which should solve the task.
        # This network has one GCN layer with sigmoid nonlinearity
        # which
        #
        # alpha[i] = sigma(f * (X[i] - 0.5)), which implements the identity
        # beta[i]  = sigma((X[i] + 2 * A[i, :] * X - 4.5) * f), which implements
        #            the lower bound for nodes being alive in the next iteration
        # gamma[i] = sigma((X[i] + 2 * A[i, :] * X - 7.5) * f), which implements
        #            the upper bound for nodes being alive in the next iteration
        #
        # in all cases, f > 1 is a factor to make the sigmoid more close to
        # the heaviside function
        #
        # based on these three features, our output is simply:
        # delta[i] = beta[i] - gamma[i] - alpha[i]
        net = gen.GEN(1, 1, 3, nonlin = torch.nn.Sigmoid())

        f = 30.
        # construct alpha feature
        net._layers[0]._W.weight[0, 0] = 1. * f
        net._layers[0]._U.weight[0, 0] = 0.
        net._layers[0]._W.bias[0]      = -0.5 * f
        # construct beta feature
        net._layers[0]._W.weight[1, 0] = 1. * f
        net._layers[0]._U.weight[1, 0] = 2. * f
        net._layers[0]._W.bias[1]      = -4.5 * f
        # construct gamma feature
        net._layers[0]._W.weight[2, 0] = 1. * f
        net._layers[0]._U.weight[2, 0] = 2. * f
        net._layers[0]._W.bias[2]      = -7.5 * f
        # set remaining values to zero
        net._layers[0]._U.bias[:] = 0.
        net._layers[0]._V.weight[:,:] = 0.
        net._layers[0]._V.bias[:] = 0.
        # construct output
        net._node.weight[0, 0] = -1.
        net._node.weight[0, 1] = +1.
        net._node.weight[0, 2] = -1.
        net._node.bias[0]      = 0.

        # test this network on a few example grids and ensure that the
        # predictive loss is close to zero
        for r in range(100):
            X = torch.rand(m, 1)
            X[X < 1. - p] = 0.
            X[X >= 1. - p] = 1.
            # apply the net
            delta_pred, _ = net(A, X)
            # generate the ground truth
            # A node is alive in the next generation if 2 * the sum of
            # all alive nodes plus the node state itself is between
            # 5 and 7
            Xnext = (X + 2 * torch.mm(A, X)).squeeze(1)
            Xnext[Xnext < 4.5] = 0.
            Xnext[Xnext > 7.5] = 0.
            Xnext[Xnext > 4.] = 1.
            # the true change vector is Xnext - X
            delta_true = (Xnext - X.squeeze(1))
            # compute the loss
            loss = loss_fun(X.squeeze(1), delta_pred, delta_true)
            # assert small loss
            if loss.item() > 1E-2:
                print('got a nontrivial loss of %g for the following grid:' % loss.item())
                print(torch.reshape(X, (grid_size, grid_size)).detach().numpy())
                print('We should have gotten the edit values:')
                print(torch.reshape(delta_true, (grid_size, grid_size)).detach().numpy())
                print('But we got the edit values:')
                print(torch.round(torch.reshape(delta_pred, (grid_size, grid_size))).detach().numpy())
            self.assertTrue(loss.item() < 1E-2)

        # set up a graph edit network for this task. A single GCN layer suffices
        # because we only need to integrate information from our immediate neighbors.
        # However, we do not expect our network to find the features we manually
        # constructed above, such that we use a larger number of neurons.
        dim_hid = 64
        net = gen.GEN(1, 1, dim_hid)

        # instantiate optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=1E-3, weight_decay=1E-3)

        # start training
        loss_threshold = 1E-3
        epochs = 1
        max_epochs = 50000
        loss_avg = 100.
        loss_avgs = []
        while loss_avg > loss_threshold and epochs < max_epochs:
            optimizer.zero_grad()
            # generate a new 'game of life' grid with random population
            # pattern
            X = torch.rand(m, 1)
            X[X < 1. - p] = 0.
            X[X >= 1. - p] = 1.

            # simulate that grid for at most max_steps
            simulation_loss = 0.
            for t in range(max_steps):
                # predict the node state changes with the current network
                # parameters
                delta_pred, _ = net(A, X)
                # generate the ground truth
                # A node is alive in the next generation if 2 * the sum of
                # all alive nodes plus the node state itself is between
                # 5 and 7
                Xnext = (X + 2 * torch.mm(A, X)).squeeze(1)
                Xnext[Xnext < 4.5] = 0.
                Xnext[Xnext > 7.5] = 0.
                Xnext[Xnext > 4.] = 1.
                # the true change vector is Xnext - X
                delta_true = (Xnext - X.squeeze(1))
                # compute the loss
                loss = loss_fun(X.squeeze(1), delta_pred, delta_true)
                simulation_loss += loss.item()
                # compute the gradient
                loss.backward()
                # stop the simulation if nothing changes anymore
                if torch.sum(torch.abs(delta_true)) < 1E-3:
                    break
            # perform an optimizer step
            optimizer.step()
            # compute a new moving average over the loss
            loss_avg = loss_avg * 0.9 + 0.1 * simulation_loss
            loss_avgs.append(loss_avg)
            if(epochs % 20 == 0):
                print('loss avg after %d epochs: %g' % (epochs, loss_avg))
            epochs += 1

        self.assertTrue(epochs < max_epochs)

    def test_edge_filtering(self):
        if _SKIP_LONG_TESTS:
            return
        print('edge filter test')
        # test whether a graph edit network can learn to predict very sparse
        # edge changes in big graphs
        # To test that, we simulate a graph connected in a circle where a
        # single additional edge exists connecting some node i to i+2.
        # This edge should then move to (i+1, i+3), i.e. we want only one
        # edge deletion and one edge insertion
        m = 100 # graph size
        T = 10  # simulation time

        # set up a graph edit network for this task. We need two GCN layers
        # because we consider a neighborhood range of 2.
        # A few neurons should suffice, though.
        dim_hid = 16
        net = gen.GEN(num_layers = 2, dim_in = 1, dim_hid = dim_hid, filter_edge_edits = True)

        # instantiate optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=1E-3, weight_decay=1E-5)
        # and loss
        loss_fun  = gen.GEN_loss()

        # start training
        loss_threshold = 1E-3
        epochs = 1
        max_epochs = 10000
        loss_avg = 100.
        loss_avgs = []
        start = time.time()
        while loss_avg > loss_threshold and epochs < max_epochs:
            optimizer.zero_grad()
            # generate a cyclic graph of size m, where all node attributes
            # are one
            X = torch.ones(m, 1)
            A = torch.zeros(m, m)
            for i in range(m-1):
                A[i, i+1] = 1.
            A[-1, 0] = 1.
            # add the 'jump' edge at a random index
            i = random.randrange(m)
            j = (i+2)%m
            A[i, j] = 1.
            # start simulation
            simulation_loss = 0.
            for t in range(T):
                # predict the node and edge changes with the network
                delta, Epsilon, edge_filters = net(A, X)
                # set up the 'true' changes
                delta_true   = torch.zeros(m)
                Epsilon_true = torch.zeros(m, m)
                Epsilon_true[i, j] = -1.
                i = (i+1)%m
                j = (j+1)%m
                Epsilon_true[i, j] = +1.
                # compute the loss
                loss = loss_fun(delta, Epsilon, delta_true, Epsilon_true, A, edge_filters)
                simulation_loss += loss.item()
                # compute the gradient
                loss.backward()
                # update the adjacency matrix
                A += Epsilon_true
            # perform an optimizer step
            optimizer.step()
            # compute a new moving average over the loss
            loss_avg = loss_avg * 0.9 + 0.1 * simulation_loss
            loss_avgs.append(loss_avg)
            if(epochs % 20 == 0):
                print('loss avg after %d epochs (time: %g): %g' % (epochs, time.time() - start, loss_avg))
                print('filtered outgoing edges: %d' % torch.sum(edge_filters[:, 0] > 0.).item())
                print('filtered incoming edges: %d' % torch.sum(edge_filters[:, 1] > 0.).item())
            epochs += 1

        self.assertTrue(epochs < max_epochs)

if __name__ == '__main__':
    unittest.main()
