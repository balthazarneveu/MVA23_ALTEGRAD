"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""

import torch
import torch.nn as nn

# Learn to add! (... add to learn...)

# => WARNING: this is not a joke,
# neural networks inherently perform MAD operation (multiply & add operation).
# multiplication is performed with weights, additions with biases
# but what matters here is the ability to add accross neurons!

# Learning to multiply digits would be much harder

# A "degenerate" case seem to exist to solve the "learn to add" task
# Provided that the embedding dimension is large enough.
# Embeddings can encode a encode a digit in a very simple form.
# [1, 1, 1, 1, 0, 0 ... 0] to encode 4=1+1+1+1 for instance
# $x = sum_{1<=j<=x}(1)$
# FC1 can simply represent the identity function...
# tanh(0) = 0
# tanh(1) = 0.76
# Then all digits are summed up in the pooling layer
# FC2 simply has to divide by 0.76 and we are done!


class DeepSets(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, output_dim: int = 1):
        """Deep set model

        Args:
            input_dim (int): number of possible digits
            embedding_dim (int): vector dimension for each digit embedding (h1)
            hidden_dim (int): hidden dimension for the 2 FC layers (h2)
            output_dim(int): output dimension, 1 by default.
        """
        super(DeepSets, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)  # $phi(x)$
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # $rho(sum(phi(x)))$
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict an arithmetic operation on the "set" x
        Args:
            x (torch.Tensor): [N, L] taking a finite amount of values (input_dim)

        Returns:
            torch.Tensor: prediction [N]

        Task 3
        ======
        ```text
        Implement the DeepSets architecture
        - an embedding layer which projects each digit to a latent space of dimensionality h1
        - a fully-connected layer with h2 hidden units followed by a tanh activation function
        - a sum aggregator which computes the sum of the elements of each set
        - a fully-connected layer with 1 unit since the output of the model needs to be a scalar
        (i.e., the prediction of the sum of the digits contained in the set)
        ```

        Implementation details
        ======================
        - x as a vector [N, L] with values
        - Embed the integers  -> [N, L, embedding_dim]
        - FC1: [N, L, embedding_dim] -> [N, L, hidden_dim]
        - Non linearity (tanh) + Sum pooling: [N, L, hidden_dim] -> [N, hidden_dim]
        - FC2: [N, hidden_dim] -> [N, 1]

        """

        # Task 3
        x = self.embedding(x)  # [N, L] -> [N, L, embedding_dim]
        x = self.tanh(self.fc1(x))  # FC1+Non linearity -> [N, L, hidden_dim]
        x = torch.sum(x, dim=1)  # pooling=sum -> [N, L]
        x = self.fc2(x)  # FC2 -> [N, 1]
        return x.squeeze()


class LSTM(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, output_dim: int = 1):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # num_layers=1, bidirectional=False
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LSTM on the "set" x

        Args:
            x (torch.Tensor): [N, L] (batch first)

        Returns:
            torch.Tensor: prediction [N]
        Task 4:
        =======

        ```text
        Implement the LSTM architecture
        - an embedding layer which projects each digit
        to a latent space of dimensionality h1
        - an LSTM layer with h2 hidden units
        - a fully-connected layer which takes the LSTM hidden state output
        for the last time step as input and outputs a scalar (out_dim=1)
        ```
        h1 = embedding_dim
        h2 = hidden_dim

        Note on LSTM use:
        =================
        extract x = all hidden states for the last iteration (last word)
        Here num_layers=1 and bidirectional=False -> 1*1*hidden_dim
        https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        """
        # Task 4
        x = self.embedding(x)  # [N, L, embedding_dim]
        _, x, _ = self.lstm(x)  # -> [N, 1*1*hidden_dim] (batch_first=True)
        x = self.fc(x)  # [N, hidden_dim] -> [N, 1]

        return x.squeeze()
