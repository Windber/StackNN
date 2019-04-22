import torch.nn as nn
import numpy as np
import torch
class GRUController(nn.Module):
    def __init__(self, config_dict):
        
        super().__init__()
        self.hidden = None
        self.params = config_dict
        
        controller_input_size = self.input_size + self.read_size
        
        self.rnn = nn.GRUCell(controller_input_size, self.hidden_size)
        self.fc_nargs = nn.Linear(self.hidden_size, self.n_args)
        self.sigmoid = nn.Sigmoid()
        self.fc_v1 = nn.Linear(self.hidden_size, self.read_size)
        self.fc_v2 = nn.Linear(self.hidden_size, self.read_size)
        self.fc_o = nn.Linear(self.hidden_size, self.output_size)
        self.tanh = nn.Tanh()
    def init(self):
        hidden_shape = (self.batch_size, self.hidden_size)
        self.hidden = torch.zeros(hidden_shape)
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
    def forward(self, x, r):
        """
        Computes an output and data structure instructions using a
        single linear layer.

        :type x: Variable
        :param x: The input to this Controller

        :type r: Variable
        :param r: The previous item read from the neural data structure

        :rtype: tuple
        :return: A tuple of the form (y, (v, u, d)), interpreted as
            follows:
                - output y
                - pop a strength u from the data structure
                - push v with strength d to the data structure
        """
        self.hidden = self.rnn(torch.cat([x, r], 1), self.hidden)
        output = self.tanh(self.fc_o(self.hidden))
        v1 = self.tanh(self.fc_v1(self.hidden))
        v2 = self.tanh(self.fc_v2(self.hidden))
        nargs = self.sigmoid(self.fc_nargs(self.hidden))
        instructions = torch.split(nargs, list(np.ones(self.n_args, dtype=np.int32)), dim=1)
        
        return output, v1, v2, instructions

    
