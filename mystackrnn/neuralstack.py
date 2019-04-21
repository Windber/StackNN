import torch.nn as nn
from abc import ABCMeta, abstractmethod
import torch
class PDAStruct(nn.Module, metaclass=ABCMeta):
    def __init__(self, embedding_size):
        super(PDAStruct, self).__init__()
        self._embedding_size = embedding_size
        
        self.contents = Variable(torch.FloatTensor(0))
        self.strengths = Variable(torch.FloatTensor(0))
        #self._readcontent = None
    def forward(self, u, d1, d2, v1, v2, r=None):
        self.push(d1, d2, v1, v2)
        readcontent = self.read(u)
        self.pop(u)
        return readcontent
    @abstractmethod
    def _init_Struct(self, batch_size):
        raise NotImplemented("Missing implementation for _init_Struct")
    @abstractmethod
    def pop(self, strength):
        raise NotImplementedError("Missing implementation for pop")

    @abstractmethod
    def push(self, strength1, strength2, value1, value2):
        raise NotImplementedError("Missing implementation for push")

    @abstractmethod
    def read(self, strength):
        raise NotImplementedError("Missing implementation for read")

    @property
    def read_strength(self):
        return 1.