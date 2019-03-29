'''
@author: lenovo
'''
from .simple import Operation
from .simple import bottom, bottom_to_top, top, top_to_bottom
from .base import Struct
from abc import ABCMeta, abstractmethod

class MediumStruct(Struct, metaclass=ABCMeta):
    def __init__(self, batch_size, embedding_size, k=None):
        super(MediumStruct, self).__init__(batch_size, embedding_size)
        operations = [Operation.push, Operation.pop]
        
        #self._reg_trackers = [None for _ in operations]
        self._read_strength = k
        
        self._values = []
        self._strengths = []
    
    def init_contents(self, xs):
        # Becasue its action always "push first and pop then"
        # So dont need this function
        pass
    
    def __len__(self):
        return len(self._values)

    @abstractmethod
    def _pop_indices(self):
        raise NotImplementedError("Missing implementation for _pop_indices")
    
    @abstractmethod
    def _push_index(self):
        raise NotImplementedError("Missing implementation for _push_index")
    @abstractmethod
    def _read_indices(self):
        raise NotImplementedError("Missing implementation for _read_indices")
    @property
    def read_strength(self):
        return self._read_strength
    
    def push(self, strength1, strength2, value1, value2):
        pass
    def read(self, strength):
        pass

        
class Stack(MediumStruct):
    def _pop_indices(self):
        return top_to_bottom(len(self))
    
    def _push_index(self):
        return top(len(self))
    
    def _read_indices(self):
        return top_to_bottom(len(self))
    
class Queue(MediumStruct):
    def _pop_indices(self):
        return bottom_to_top(len(self))
    
    def _push_index(self):
        return top(len(self))
    
    def _read_indices(self):
        return bottom_to_top(len(self))
    
    