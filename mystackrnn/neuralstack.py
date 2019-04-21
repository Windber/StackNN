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
class PDASimpleStruct(PDAStruct, metaclass=ABCMeta):
    def __init__(self, embedding_size, k=None):
        super(PDASimpleStruct, self).__init__(embedding_size)
        operations = [Operation.push, Operation.pop]
        
        self._reg_trackers = [None for _ in operations]
        self._read_strength = k
        
        self._values = None
        self._strengths = None
    
    def init_contents(self, xs):
        # Becasue its action always "push first and pop then"
        # So dont need this function
        self._values = list()
        self._strengths = list()
    
    def _init_Struct(self, batch_size):
        #self._read = Variable(torch.zeros([batch_size, self.embedding_size]))
        self._zeros = Variable(torch.zeros(batch_size))
        self.batch_size = batch_size
        self._actual = torch.zeros(batch_size, 1)
        self.init_contents(None)
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
        self._track_reg(strength1 + strength2, Operation.push)
        self._actual += (strength1 + strength2)

        push_index = self._push_index()
        self._values.insert(push_index, value1)
        self._strengths.insert(push_index, strength1)
        self._values.insert(push_index+1, value2)
        self._strengths.insert(push_index+1, strength2)
        
    def read(self, u):
        summary = Variable(torch.zeros([self.batch_size, self._embedding_size]))
        strength_used = Variable(torch.zeros(self.batch_size, 1))
        for i in self._read_indices():
            summary += self._values[i] * torch.min(self._strengths[i], torch.max(u - strength_used))
            strength_used += self._strengths[i]
        return summary
    
    def pop(self, u):
        self._track_reg(u, Operation.pop)
        self._actual = torch.max(torch.zeros(self.batch_size, 1), self._actual - u)
        strengths_used = Variable(torch.zeros(self.batch_size, 1))
        for i in self._pop_indices():
            tmp = self._strengths[i]
            # Maybe cant autograd when backwards because of in-place action on self._strengths
            self._strengths[i] = self._strengths[i] - torch.min(self._strengths[i], torch.max(torch.zeros(self.batch_size, 1), u - strengths_used))
            strengths_used += tmp
    def set_reg_tracker(self, reg_tracker, operation):
        self._reg_trackers[operation] = reg_tracker

    def _track_reg(self, strength, operation):
        reg_tracker = self._reg_trackers[operation]
        if reg_tracker is not None:
            reg_tracker.regularize(strength)
    

    def print_summary(self, batch):
        """
        Prints self._values and self._strengths to the console for a
        particular batch.

        :type batch: int
        :param batch: The number of the batch to print information for

        :return: None
        """
        if batch < 0 or batch >= self.batch_size:
            raise IndexError("There is no batch {}.".format(batch))

        print("t\t|Strength\t|Value")
        print("\t|\t\t\t|")

        for t in reversed(list(range(len(self)))):
            #v_str = to_string(self._values[t][batch, :])
            v_str = to_string(self._values[t])
            #v_str = to_string(self._values[t][batch])
            s = self._strengths[t][batch].data.item()
            print(("{}\t|{:.4f}\t\t|{}".format(t, s, v_str)))

    def log(self):
        """
        Prints self._values and self._strengths to the console for all
        batches.

        :return: None
        """
        for b in range(self.batch_size):
            print(("Batch {}:".format(b)))
            self.print_summary(b)
    @property
    def actual(self):
        return self._actual
class PDAStack(PDASimpleStruct):
    def _pop_indices(self):
        return top_to_bottom(len(self))
    
    def _push_index(self):
        return top(len(self))
    
    def _read_indices(self):
        return top_to_bottom(len(self))
    
class PDAQueue(PDASimpleStruct):
    def _pop_indices(self):
        return bottom_to_top(len(self))
    
    def _push_index(self):
        return top(len(self))
    
    def _read_indices(self):
        return bottom_to_top(len(self))
