from typing import List

import torch.nn as nn
import torch.nn.functional as F
import torch
from abc import abstractmethod, ABC

from numpy.typing import ArrayLike


class BaseModel(torch.nn.Module, ABC):
    def __init__(self, input_dim: int, output_dim: int):
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def loss(self, pred, actual):
        pass


class LogisticRegressionModel(BaseModel):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegressionModel, self).__init__(input_dim, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def loss(self, pred, actual):
        return F.nll_loss(pred, actual)


class BasicNeuralNet(BaseModel, ABC):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: int = 0.3,
                 batch_normalize: bool = True, activation_function: str = 'relu'):
        super(BasicNeuralNet, self).__init__(input_dim, output_dim)
        self.dropout = dropout
        self.batch_normalize = batch_normalize
        self.num_hidden_layers = len(hidden_dims)

        if activation_function == 'relu':
            self.activation = F.relu
        else:
            raise Exception(f'Currently only "relu" activation is supported but was provided {activation_function}')

        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.fcs.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(self.num_hidden_layers - 1):
            self.fcs.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.fcs.append(nn.Linear(hidden_dims[-1], output_dim))

        if batch_normalize:
            self.bns.append(nn.BatchNorm1d(hidden_dims[0]))
            for i in range(self.num_hidden_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_dims[i + 1]))

    def _activations_and_conditionals(self, x, i):
        if self.batch_normalize:
            x = self.bns[i](x)
        x = self.activation(x)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward(self, x):
        x = self.fcs[0](x)
        x = self._activations_and_conditionals(x, 0)
        for i in range(1, self.num_hidden_layers):
            x = self.fcs[i](x)
            x = self._activations_and_conditionals(x, i)
        x = self.fcs[-1](x)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def loss(self, pred, actual):
        return F.nll_loss(pred, actual)
