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
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from torch.autograd import Variable
from torch_geometric.nn import SAGEConv, GCNConv, GINConv

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

    def __init__(self, num_layers, dim_in, dim_hid, dim_vae=None, beta=1., sigma_scaling=1., nonlin=torch.nn.ReLU()):
        super(VGAE, self).__init__()
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError('The number of layers must be a natural number but was %s' % str(num_layers))
        self.num_layers = num_layers
        self.dim_in = dim_in
        if isinstance(dim_hid, list):
            if len(dim_hid) != num_layers:
                raise ValueError(
                    'If a hidden dimensionality for each layer is specified, the number of node dimensionalities must be exactly self._num_layers = %d, but was %d' % (
                    num_layers, len(dim_hid)))
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
            self.layers_.append(pytorch_graph_edit_networks.GCN(self.dim_hid[l - 1], self.dim_hid[l]))
        # set up the final encoding layer
        self.enc_ = torch.nn.Linear(self.dim_hid[-1], 2 * self.dim_vae)
        # set up bias for the outer product
        self.bias_ = torch.nn.Linear(1, 1)
        self.nonlin = nonlin

    def forward(self, A, X=None):
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
        Mu = self.enc_(X)[:, :self.dim_vae]
        # compute outer product
        B = self.bias_(torch.mm(Mu, Mu.t()).unsqueeze(2)).squeeze(2)
        # post-process: set diagonal to zero, all entries > 0 to 1 and all
        # other entries to zero.
        B = B - torch.diag(torch.diag(B))
        B[B > 1E-3] = 1.
        B[B <= 1E-3] = 0.
        return B

    def compute_loss(self, A, B, X=None, verbose=False):
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
        Sigma = torch.exp(0.5 * Logvar)
        # construct a random code
        if self.sigma_scaling > 0.:
            Z = torch.randn(Mu.shape[0], self.dim_vae).to(Mu.device) * Sigma * self.sigma_scaling + Mu
        else:
            Z = Mu
        # compute outer product
        Bpred = self.bias_(torch.mm(Z, Z.t()).unsqueeze(2)).squeeze(2)
        # set diagonal to zero
        Bpred = Bpred - torch.diag(torch.diag(Bpred))
        # compute binary crossentropy between B and Bpred
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Bpred.flatten(), B.flatten(), reduction='sum')

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

# Added by Daniele Grattarola

class GraphGruGCN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(GraphGruGCN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer

        # gru weights
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []

        for i in range(self.n_layer):
            if i == 0:
                self.weight_xz.append(GCNConv(input_size, hidden_size, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, bias=bias))
                self.weight_xr.append(GCNConv(input_size, hidden_size, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, bias=bias))
                self.weight_xh.append(GCNConv(input_size, hidden_size, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, bias=bias))
            else:
                self.weight_xz.append(GCNConv(hidden_size, hidden_size, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, bias=bias))
                self.weight_xr.append(GCNConv(hidden_size, hidden_size, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, bias=bias))
                self.weight_xh.append(GCNConv(hidden_size, hidden_size, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, bias=bias))

    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size())
        for i in range(self.n_layer):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
            #         out = self.decoder(h_t.view(1,-1))
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i - 1], edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i - 1], edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i - 1], edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        #         out = self.decoder(h_t.view(1,-1))

        out = h_out
        return out, h_out


class InnerProductDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid, dropout=0.):
        super(InnerProductDecoder, self).__init__()

        self.act = act
        self.dropout = dropout

    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)


class VGRNN(nn.Module):
    def __init__(self, dim_in, dim_hid, num_layers, eps=1e-10, bias=True):
        super(VGRNN, self).__init__()

        self.dim_in = dim_in
        self.eps = eps
        self.dim_hid = dim_hid
        self.z_dim = dim_hid
        self.num_layers = num_layers

        self.phi_x = nn.Sequential(nn.Linear(dim_in, dim_hid), nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(dim_hid, dim_hid), nn.ReLU())

        self.enc = GCNConv(dim_hid + dim_hid, dim_hid)
        self.enc_mean = GCNConv(dim_hid, dim_hid)
        self.enc_std = GCNConv(dim_hid, dim_hid)

        self.prior = nn.Sequential(nn.Linear(dim_hid, dim_hid), nn.ReLU())
        self.prior_mean = nn.Sequential(nn.Linear(dim_hid, dim_hid))
        self.prior_std = nn.Sequential(nn.Linear(dim_hid, dim_hid), nn.Softplus())

        self.rnn = GraphGruGCN(dim_hid + dim_hid, dim_hid, num_layers, bias)

    def forward(self, x, edge_idx, hidden_in=None):
        if hidden_in is None:
            h = Variable(torch.zeros(self.num_layers, x.shape[-2], self.dim_hid))
        else:
            h = Variable(hidden_in)

        phi_x_t = self.phi_x(x)

        # encoder
        enc = self.enc(torch.cat([phi_x_t, h[-1]], 1), edge_idx)
        enc_mean = self.enc_mean(enc, edge_idx)
        enc_std = F.softplus(self.enc_std(enc, edge_idx))

        # prior
        prior = self.prior(h[-1])
        prior_mean = self.prior_mean(prior)
        prior_std = self.prior_std(prior)

        # sampling and reparameterization
        z = self._reparameterized_sample(enc_mean, enc_std)
        phi_z = self.phi_z(z)

        # decoder
        dec = self.dec(z)

        # recurrence
        _, h = self.rnn(torch.cat([phi_x_t, phi_z], 1), edge_idx, h)

        return dec, enc_mean, enc_std, prior_mean, prior_std, h

    def compute_loss(self, dec, enc_mean, enc_std, prior_mean, prior_std, target_adj):
        nnodes = target_adj.shape[-1]
        enc_mean_sl = enc_mean[0:nnodes, :]
        enc_std_sl = enc_std[0:nnodes, :]
        prior_mean_sl = prior_mean[0:nnodes, :]
        prior_std_sl = prior_std[0:nnodes, :]
        dec_sl = dec[0:nnodes, 0:nnodes]

        kld_loss = self._kld_gauss(enc_mean_sl, enc_std_sl, prior_mean_sl, prior_std_sl)
        nll_loss = self._nll_bernoulli(dec_sl, target_adj)

        return kld_loss + nll_loss

    def dec(self, z):
        outputs = InnerProductDecoder()(z)
        return outputs

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_()
        eps1 = Variable(eps1)
        return eps1.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element = (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                       (torch.pow(std_1 + self.eps, 2) + torch.pow(mean_1 - mean_2, 2)) /
                       torch.pow(std_2 + self.eps, 2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)

    def _kld_gauss_zu(self, mean_in, std_in):
        num_nodes = mean_in.size()[0]
        std_log = torch.log(std_in + self.eps)
        kld_element = torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) -
                                           torch.pow(torch.exp(std_log), 2), 1))
        return (-0.5 / num_nodes) * kld_element

    def _nll_bernoulli(self, logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        posw = float(temp_size * temp_size - temp_sum) / (temp_sum + 1e-7)
        norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(input=logits,
                                                          target=target_adj_dense,
                                                          pos_weight=posw,
                                                          reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0, 1])
        return - nll_loss
