import torch.nn as nn
from abc import ABCMeta, abstractmethod
import torch
from structs.simple import SimpleStruct 
class NeuralMemory(SimpleStruct, metaclass=ABCMeta):
    def __init__(self, batch_size, read_size, k=None):
        super().__init__(batch_size, read_size)
        self._values = None
        self._strengths = None
        self._actual = None
        self.batch_size = batch_size
        self.read_size = read_size 
    def forward(self, u, d1, d2, v1, v2, r=None):
        self.push(d1, d2, v1, v2)
        readcontent = self.read(u)
        self.pop(u)
        return readcontent
    def init(self, batch_size=None):
        if batch_size:
            self.batch_size = batch_size
        self._values = list()
        self._strengths = list()
        self._actual = torch.zeros(self.batch_size, 1)
    def push(self, strength1, strength2, value1, value2):
        self._actual = self._actual + strength1 + strength2
        self._values.append(value1)
        self._strengths.append(strength1)
        self._values.append(value2)
        self._strengths.append(strength2)
        
    def read(self, u):
        summary = torch.zeros([self.batch_size, self.read_size])
        strength_used = torch.zeros(self.batch_size, 1)
        for i in self._read_indices():
            summary += self._values[i] * torch.min(self._strengths[i], torch.max(u - strength_used))
            strength_used += self._strengths[i]
        return summary
    
    def pop(self, u):
        self._actual = torch.max(torch.zeros(self.batch_size, 1), self._actual - u)
        strength_used = torch.zeros(self.batch_size, 1)
        for i in self._pop_indices():
            tmp = self._strengths[i]
            self._strengths[i] = self._strengths[i] - torch.min(self._strengths[i], torch.max(torch.zeros(self.batch_size, 1), u - strength_used))
            strength_used += tmp

class PDAStack(NeuralMemory):
    def _pop_indices(self):
        return list(range(len(self._strengths)-1, -1, -1))
    
    def _push_index(self):
        return len(self._strengths)
    
    def _read_indices(self):
        return list(range(len(self._strengths)-1, -1, -1))
    
class PDAQueue(NeuralMemory):
    def _pop_indices(self):
        return list(range(0, len(self._strengths)))
    
    def _push_index(self):
        return len(self._strengths)
    
    def _read_indices(self):
        return list(range(0, len(self._strengths)))
