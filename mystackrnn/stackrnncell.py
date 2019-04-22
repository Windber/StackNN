import torch.nn as nn
from mystackrnn.neuralcontroller import GRUController
from mystackrnn.neuralstack import NeuralStack
import torch
class StackRNNCell(nn.Module):

    def __init__(self, config_dict):
        super().__init__()
        self.params = config_dict
        
        self._read = None
        self._u = None
        self._s1 = None
        self._s2 = None
        self._v1 = None
        self._v2 = None

        self.controller = GRUController(config_dict)
        self.struct = NeuralStack(config_dict)
        
    

    def init(self):
        self.struct.init()
        
        self.controller.init()

        self._read = torch.zeros(self.batch_size, self.read_size)
        

    def forward(self, inp):
        
        o, self._v1, self._v2, (self._s1, self._s2, self._u)= self.controller(inp, self._read)
        
        self._read = self.struct(self._u, self._s1, self._s2, self._v1, self._v2)
        
        return o
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return super().__getattr__(name)
if __name__ == "__main__":
    from mystackrnn.profile import config_dyck2
    ns = NeuralStack(config_dyck2)
    gc = GRUController(config_dyck2)
    print(ns, gc)


