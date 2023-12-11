"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool

# Decoder


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes, dropout=0.2):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        self.fc = nn.ModuleList()
        self.fc.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.LayerNorm(hidden_dim),
                                     nn.Dropout(dropout)
                                     ))

        for i in range(1, n_layers):
            self.fc.append(nn.Sequential(nn.Linear(hidden_dim*i, hidden_dim*(i+1)),
                                         nn.ReLU(),
                                         nn.LayerNorm(hidden_dim*(i+1)),
                                         nn.Dropout(dropout)
                                         ))

        self.fc_proj = nn.Linear(hidden_dim*n_layers, n_nodes*n_nodes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Task 10
        for i in range(self.n_layers):
            x = self.fc[i](x)
        x = self.fc_proj(x)
        adj = x.reshape(-1, self.n_nodes, self.n_nodes)
        adj = 0.5*(adj + adj.transpose(1, 2))
        # (A + A^T)/2 => trick make A symmetric
        # sigmoid will be added in the loss function (check VariationalAutoEncoder.decode)

        return adj

# Encoder


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.mlps = torch.nn.ModuleList()
        self.mlps.append(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU()
                                       ))

        for layer in range(n_layers-1):
            self.mlps.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU()
                                           ))

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, adj, x, idx):
        # Task 8
        # implement H(1) and H(2)
        for i in range(self.n_layers):
            x = adj @ x
            x = self.mlps[i](x)
        # Readout
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(int(torch.max(idx))+1, x.size(1), device=x.device, requires_grad=False)
        out = out.scatter_add_(0, idx, x)

        out = self.fc(out)
        return out


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def reparameterize(self, mu, logvar, eps_scale=1.):
        """Trick to backpropagate through the sampling process

        Question: How to learn in presence of stochastic variables?
        Answer: Reparameterization trick (Kingma and Welling, 2013)

        Sample from a standard normal distribution
        - and shift by mu
        - and scale by "std"

        https://arxiv.org/abs/1312.6114
        """
        if self.training:
            # We do not parameterize directly with sigma because sigma must be positive
            # Instead we train on logvar where sigma=exp(0.5*logvar) => no constraint on logvar
            std = logvar.mul(0.5).exp_()  # sigma
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu):
        adj = self.decoder(mu)
        adj = torch.sigmoid(adj)
        adj = adj * (1 - torch.eye(adj.size(-2), adj.size(-1), device=adj.device))
        return adj

    def loss_function(self, adj, x, idx, y, beta=0.05):
        """Loss function for the variational autoencoder
        Part of the nn.module as it needs acess to forward pass variables
        """
        x_g = self.encoder(adj, x, idx)

        # Task 9

        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)

        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)

        triu_idx = torch.triu_indices(self.n_max_nodes, self.n_max_nodes)
        recon = F.binary_cross_entropy_with_logits(
            adj[:, triu_idx[0, :], triu_idx[1, :]], y[:, triu_idx[0, :], triu_idx[1, :]],
            reduction='sum',
            pos_weight=torch.tensor(1./0.4)
        )
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Force latent prior to look like a standard normal distribution using
        # Kulback Leibler divergence
        loss = recon + beta*kld

        return loss, recon, kld
