'''
@author: lenovo
'''
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mystackrnn.stackrnncell import StackRNNCell
from mystackrnn.neuralcontroller import GRUController
from mystackrnn.neuralstack import NeuralStack
import pandas as pd
import random
import time
class StackRNNTask:
    def __init__(self, config_dict):
        self.params = config_dict
        
        self.trainx, self.trainy, self.testxl, self.testyl = self.get_data()
        
        self.model = StackRNNCell(config_dict)
        
        self.loss_classify = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = torch.optim.RMSprop(self.model.parameters())
        if self.load:
            self.state, self.minloss, self.maxaccuracy = torch.load(self.load_path)
            self.model.load_state_dict(self.state)
        else:
            self.state = None
            self.minloss = 1e3
            self.maxaccuracy = 0.
    def experiment(self):
        if not self.onlytest:
            self.train()
        self.test()
        
    def test(self):
        print("Test stage:")
        for testx, testy in zip(self.testxl, self.testyl):
            eloss, eacc = self.perepoch(testx, testy, 1, False)
            
    def train(self):
        print("Train stage:")
        for e in range(self.epochs):
            e += 1
            eloss, eacc = self.perepoch(self.trainx, self.trainy, e, True)
            if eloss < self.minloss and eacc > self.maxaccuracy:
                self.state = self.model.state_dict()
                self.minloss = eloss.item()
                self.maxaccuracy = eacc
                torch.save([self.state, self.minloss, self.maxaccuracy], 
                           self.saved_path_prefix + time.strftime("%d%H%M") +  "_%.2f_%.2f" % (self.minloss, self.maxaccuracy))
                
    def perepoch(self, ex, ey, e, istraining):
        samples = ex.shape[0]
        bsize = self.batch_size
        batchs = samples // bsize
        queue = torch.randperm(samples)
        eloss = 0
        etotal = 0
        ecorrect = 0
        for b in range(batchs):
            bstart = b * bsize
            bend = (b + 1) * bsize
            xs = ex[queue[bstart: bend]]
            ys = ey[queue[bstart: bend]]
            self.model.init()
            bloss, bcorrect, btotal = self.perbatch(xs, ys, b, istraining)
            eloss += bloss
            etotal += btotal
            ecorrect += bcorrect
        eavgloss = eloss / etotal
        eaccuracy = ecorrect / etotal
        print("Epoch %d Loss: %f Accuracy: %f" % (e, eavgloss, eaccuracy))
        return eavgloss, eaccuracy
    
    def perbatch(self, bx, by, b, istraining):
        bsize = self.batch_size
        steps = bx.shape[1]
        btotal = bsize * steps
        yp = torch.zeros((bsize, steps, self.output_size))
        for i in range(steps):
            outp = self.model(bx[:, i, :])
            yp[:, i, :] = outp
        
        _, yp_index = torch.topk(yp, 1, dim=2)
        yp_index = yp_index.view(yp_index.shape[0], yp_index.shape[1])
        bcorrect = torch.sum( yp_index == by).item()
        yp = yp.view(-1, 2)
        ys = by.view(-1)
        bloss = self.loss_classify(yp, ys)
        if istraining:
            self.optimizer.zero_grad()
            bloss.backward()
            self.optimizer.step()
        if self.verbose:
            print("Batch %d Loss: %f Accuracy: %f" % (b, bloss / btotal, bcorrect / btotal))
        return bloss, bcorrect, btotal
    
    def perstep(self):
        pass
    
    def get_data(self):
        trdata = pd.read_csv(self.trpath, header=None, index_col=None)
        trdx = trdata[0].values.tolist()
        trdy = trdata[1].values.tolist()
        xmap = self.alphabet
        ymap = self.classes
        trdx = [list(map(lambda x: xmap[x], s)) for s in trdx]
        
        trdy = [list(map(lambda x: ymap[x], s)) for s in trdy]
        trtx = torch.Tensor(trdx).long()
        trty = torch.Tensor(trdy).long().to(self.device)
        total = len(trdx)
        steps = len(trdx[0])
        trtx = torch.zeros(total, steps, self.input_size).scatter_(2, trtx, 1.).to(self.device)
        
        tetxlist = list()
        tetylist = list()
        for tn in range(1, self.testfile_num+1):
            tedata = pd.read_csv(self.tepath_prefix + str(tn), header=None, index_col=None)
            tedx = tedata[0].values.tolist()
            tedy = tedata[1].values.tolist()
            tedx = [list(map(lambda x: xmap[x], s)) for s in tedx]
            tedy = [list(map(lambda x: ymap[x], s)) for s in tedy]
            tetx = torch.Tensor(tedx).long()
            tety = torch.Tensor(tedy).long().to(self.device)
            total = len(tedx)
            steps = len(tedx[0])
            tetx = torch.zeros(total, steps, self.input_size).scatter_(2, tetx, 1.).to(self.device)
            tetxlist.append(tetx)
            tetylist.append(tety)
        return trtx, trty, tetxlist, tetylist
    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        else:
            return None