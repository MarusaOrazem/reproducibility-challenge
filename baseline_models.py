"""
Implements baseline models for our experiments

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

import pytorch_graph_edit_networks

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020-2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__ = 'bpaassen@techfak.uni-bielefeld.de'

class VGAE(torch.nn.Module):
    """ We implement a version of variational graph auto encoders by Kipf and
    Welling (2017). However, instead of auto-encoding, this variation here
    attempts to predict the next step of a graph, as suggested by
    Variational Graph Recurrent Neural Networks (Hajiramezanali et al., 2019).

    To make this architecture comparable with graph edit networks, we use the
    same basic architecture with a certain number of node feature layers,
    where the output adjacency matrix is computed via the outer product of
    node features.

    The loss for training is the binary crossentropy between the actual next
    adjacency matrix entries and the outer product, plus a regularization
    term that keeps the last layer of node features close to the standard
    normal distribution.

    Attributes
    ----------
    num_layers: int
        The number of GCN layers.
    dim_in: int
        The input dimensionality of node representations.
    dim_hid: int
        The hidden dimensionality of each GCN layer. Can be a single int
        or a list of ints of length num_layers.
    dim_vae: int (default = dim_hid[-1])
        The dimensionality for the VAE encoding space.
    beta: float (default = 1.)
        The regularization strength.
    sigma_scaling: float (default = 1.)
        A scaling factor for the noise on the last layer node features.
    nonlin: class torch.nn.Module (default = torch.nn.ReLU())
        The nonlinearity applied after each layer.
    layers_: class torch.nn.ModuleList
        The GCN layers to compute the node features.
    enc_: class torch.nn.Linear
        The 2*dim_vae x dim_hid[-1] final encoding layer of the VAE
        to compute mean and standard deviation of the final node features in
        the encoding space.

    """
    def __init__(self, num_layers, dim_in, dim_hid, dim_vae = None, beta = 1., sigma_scaling = 1., nonlin = torch.nn.ReLU()):
        super(VGAE, self).__init__()
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError('The number of layers must be a natural number but was %s' % str(num_layers))
        self.num_layers = num_layers
        self.dim_in     = dim_in
        if isinstance(dim_hid, list):
            if len(dim_hid) != num_layers:
                raise ValueError('If a hidden dimensionality for each layer is specified, the number of node dimensionalities must be exactly self._num_layers = %d, but was %d' % (num_layers, len(dim_hid)))
            self.dim_hid = dim_hid
        else:
            self.dim_hid = [dim_hid] * self.num_layers
        if dim_vae is None:
            self.dim_vae = self.dim_hid[-1]
        else:
            self.dim_vae = dim_vae
        self.beta = beta
        self.sigma_scaling = sigma_scaling
        # generate the GCN layers
        self.layers_ = torch.nn.ModuleList()
        # first, an input layer
        self.layers_.append(pytorch_graph_edit_networks.GCN(self.dim_in, self.dim_hid[0]))
        # then, all hidden layers up to the last one
        for l in range(1, self.num_layers):
            self.layers_.append(pytorch_graph_edit_networks.GCN(self.dim_hid[l-1], self.dim_hid[l]))
        # set up the final encoding layer
        self.enc_ = torch.nn.Linear(self.dim_hid[-1], 2 * self.dim_vae)
        # set up bias for the outer product
        self.bias_ = torch.nn.Linear(1, 1)
        self.nonlin = nonlin

    def forward(self, A, X = None):
        """ Computes the next adjacency matrix given a current adjacency matrix
        and current node features.

        Parameters
        ----------
        A: class torch.Tensor
            An m x m adjacency matrix.
        X: class torch.Tensor (default = None)
            the m x self.dim_in matrix of initial node representations.

        Returns
        -------
        B: class torch.Tensor
            An m x m next adjacency matrix.

        """
        if X is None:
            X = torch.zeros(A.shape[0], self.dim_in)

        A = A.detach()
        # apply each layer
        for l in range(self.num_layers):
            X = self.nonlin(self.layers_[l](X, A))
        # apply final layer without nonlinearity and consider only the mean
        z = self.enc_(X)[:, :self.dim_vae]
        # compute outer product
        B = self.bias_(torch.mm(X, X.t()).unsqueeze(2)).squeeze(2)
        # post-process: set diagonal to zero, all entries > 0 to 1 and all
        # other entries to zero.
        B = B - torch.diag(torch.diag(B))
        B[B > 1E-3] = 1.
        B[B <= 1E-3] = 0.
        return B

    def compute_loss(self, A, B, X = None, verbose = False):
        """ Computes the loss for predicting the adjacency matrix B from the
        adjacency matrix A and the node features X.


        Parameters
        ----------
        A: class torch.Tensor
            An m x m adjacency matrix.
        B: class torch.Tensor
            The m x m next-step adjacency matrix.
        X: class torch.Tensor (default = None)
            the m x self.dim_in matrix of initial node representations.

        Returns
        -------
        loss: class torch.Tensor
            The loss for predicting B from A and X.

        """
        if X is None:
            X = torch.zeros(A.shape[0], self.dim_in)

        A = A.detach()
        # apply each layer
        for l in range(self.num_layers):
            X = self.nonlin(self.layers_[l](X, A))
        # apply final layer without nonlinearity
        mu_and_logvar = self.enc_(X)
        # Retrieve means, log-variances and standard deviations
        Mu = mu_and_logvar[:, :self.dim_vae]
        Logvar = mu_and_logvar[:, self.dim_vae:]
        Sigma  = torch.exp(0.5 * Logvar)
        # construct a random code
        if self.sigma_scaling > 0.:
            Z = torch.randn(Mu.shape[0], self.dim_vae).to(Mu.device) * Sigma * self.sigma_scaling + Mu
        else:
            Z = Mu
        # compute outer product
        Bpred = self.bias_(torch.mm(X, X.t()).unsqueeze(2)).squeeze(2)
        # set diagonal to zero
        Bpred = Bpred - torch.diag(torch.diag(Bpred))
        # compute binary crossentropy between B and Bpred
        loss  = torch.nn.functional.binary_cross_entropy_with_logits(Bpred.flatten(), B.flatten(), reduction = 'sum')

        if verbose:
            # print(torch.stack([Bpred.flatten(), B.flatten()], 1).detach().numpy())
            print('reconstruction loss: %g' % loss.item())
        # add the regularization
        if self.beta > 0.:
            Mu2 = torch.pow(Mu, 2)
            Sigma2 = torch.exp(Logvar)
            regul_loss = self.beta * torch.sum(Mu2 + Sigma2 - Logvar - 1)
            if verbose:
                print('regularization loss: %g' % regul_loss.item())
            loss = loss + regul_loss
        return loss

