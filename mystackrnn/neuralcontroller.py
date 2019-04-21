from controllers.base import SimpleStructController
import torch.nn as nn
import numpy as np
class PDAGRUSimpleStructController(SimpleStructController):
    def __init__(self, batch_size, input_size, hidden_size, read_size, output_size, n_args,):
        super().__init__(input_size, read_size, output_size, n_args)
        self._hidden = None
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.read_size = read_size
        self.output_size = output_size
        self.n_args = n_args
        # Create an GRU Module object
        controller_input_size = self.input_size + self.read_size
        #nn_output_size = self.n_args + self.read_size * 2 + self.output_size
        self.rnn = nn.GRUCell(controller_input_size, hidden_size)
        self.fc_nargs = nn.Linear(hidden_size, self._n_args)
        self.sigmoid = nn.Sigmoid()
        self.fc_v1 = nn.Linear(hidden_size, self._read_size)
        self.fc_v2 = nn.Linear(hidden_size, self._read_size)
        self.fc_o = nn.Linear(hidden_size, self._output_size)
        self.tanh = nn.Tanh()
    def init(self, batch_size=None):
        if batch_size:
            self.batch_size = batch_size
        hidden_shape = (batch_size, self._gru.hidden_size)
        self._hidden = torch.zeros(hidden_shape)

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
        self._hidden = self._gru(torch.cat([x, r], 1), self._hidden)
        output = self.tanh(self.fc_o(self._hidden))
        v1 = self.tanh(self.fc_v1(self._hidden))
        v2 = self.tanh(self.fc_v2(self._hidden))
        nargs = self.sigmoid(self.fc_nargs(self._hidden))
        nargs = nargs.view(nargs.shape[0], nargs.shape[1], 1)
        instructions = torch.split(nargs, list(np.ones(self.nargs, dtype=np.int32)), dim=1)
        # output, v1, v2, (s1, s2, u)
        return output, v1, v2, instructions

    
