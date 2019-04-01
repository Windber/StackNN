'''
@author: lenovo
'''
from abc import ABCMeta, abstractmethod, abstractproperty

from copy import copy, deepcopy
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from models.vanilla import PDAVanillaModel
from controllers.recurrent import PDALSTMSimpleStructController
from stacknn_utils import *
from structs.simple import Stack, PDAStack
from .base import Task
import pandas as pd
import time
class PDATask(Task, metaclass=ABCMeta):
    class Params(Task.Params):

        """
        parameters in config_dict:
            task*
        parameters in Task.Params:
            model_type*
            controller_type*
            struct_type*
            batch_size*
            clipping_norm
            criterion
            early_stopping_steps
            hidden_size*
            learning_rate*
            l2_weight
            read_size*
            time_function
            verbose
            verbosity
            load_path*
            save_path*
            test_override
            custom_initialization
        parameters in PDATask.Params:
            input_size
            output_size
            leafting_norm:gradient minmum value
            trd_path
            ted_path
        """

        def __init__(self, **kwargs):
            self.input_size = kwargs.get("input_size", 6)
            self.output_size = kwargs.get("output_size", 2)
            self.leafting_norm = kwargs.get("leafting_norm", 0.2)
            self.trd_path = kwargs.get("trd_path", None)
            self.ted_path = kwargs.get("ted_path", None)
            #del kwargs["input_size"]
            #del kwargs["output_size"]
            #del kwargs["leafting_norm"]
            super(PDATask.Params, self).__init__(**kwargs)
    def __init__(self, params):
        self.params = params
        self.model = self._init_model()
        if self.params.cuda:
            if torch.cuda.is_available():
                self.model.cuda()
                print("CUDA enabled!")
            else:
                warnings.warn("CUDA is not available.")
        # Load a saved model if one is specified.
        if self.params.load_path:
            self.model.load_state_dict(torch.load(self.load_path))
            self._has_trained_model = True
        else:
            self._has_trained_model = False
        
        self.embedding = None
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.params.learning_rate,
                                    weight_decay=self.params.l2_weight)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
        # Initialize training and testing data.
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        # Initialize various reporting hidden variables.
        self._logging = False
        self._logged_x_text = None
        self._logged_y_text = None
        self._logged_a = None
        self._logged_loss = None
        self._logged_correct = None
        self._curr_log_index = 0
        self.batch_acc = None
        
        #self.input_size = 8
        #self.output_size = 2
    def _init_model(self):
        # TODO: Should initialize controller/task here and pass it in.
        return self.params.model_type(self.params.input_size,
                                      self.params.read_size,
                                      self.params.output_size,
                                      controller_type=self.params.controller_type,
                                      struct_type=self.params.struct_type,
                                      hidden_size=self.params.hidden_size,
                                      reg_weight=self.params.reg_weight,
                                      custom_initialization=self.params.custom_initialization)
    @classmethod
    def from_config_dict(cls, config_dict):
        """Create a new task instance from a config dict."""
        if "task" not in config_dict:
            raise ValueError("Config dictionary does not contain a task.")
        if not issubclass(config_dict["task"], cls):
            raise ValueError("Invalid task type %s." % config_dict["task"])
        
        task_type = config_dict["task"]
        config_dict = copy(config_dict)
        del config_dict["task"]
        params = task_type.Params(**config_dict)
        return task_type(params)
    def _evaluate_step(self, x, y, a, j):
        raise NotImplementedError("Missing implementation for _evaluate_step")

    def run_experiment(self):
        self._print_experiment_start()
        # To do
        self.get_data()
        accuracy_best = 0
        for epoch in range(self.params.epochs):
            self.run_epoch(epoch)
            if self.batch_acc >= accuracy_best:
                accuracy_best = self.batch_acc
                if self.save_path:
                    torch.save(self.model.state_dict(), self.save_path)
        self._has_trained_model = True
        return dict(best_acc=accuracy_best, final_acc=self.batch_acc)
    
    def run_epoch(self, epoch):
        self._print_epoch_start(epoch)
        self._shuffle_training_data()
        self.train()
        self.evaluate(epoch)

    def train(self):
        """
        Trains the model given by self.model for an epoch using the data
        given by self.train_x and self.train_y.

        :return: None
        """
        if self.model is None:
            raise ValueError("Missing model.")
        if self.train_x is None or self.train_y is None:
            raise ValueError("Missing training data.")

        self.model.train()
        last_trial = len(self.train_x.data) - self.batch_size + 1
        for batch, i in enumerate(range(0, last_trial, self.batch_size)):

            # Embed x if it is not one-hot.
            if self.embedding is None:
                x = self.train_x[i:i + self.batch_size, :, :]
            else:
                xi = self.train_x[i:i + self.batch_size, :]
                x = self.embedding(xi)

            # Currently, y must be a [batch_size, num_steps] tensor.
            y = self.train_y[i:i + self.batch_size, :]

            self.model.init_model(self.batch_size)
            self._evaluate_batch(x, y, batch, True)
    def _evaluate_batch(self, x, y, name, is_batch):
        # size of x: (batch_size, characters, input_size/embedding_size)
        # size of y: (batch_size, characters, output_size)
        batch_size = x.size()[0]
        start_time = time.time()
        batch_loss = torch.zeros(1)
        batch_correct = 0
        batch_total = 0
        # size of actual_count: (batch_size, 1)
        #actual_count 
        feed_count = 0
        inp_len = x.size()[1]
        outputs_tensor = torch.Tensor()
        z_tensor = torch.Tensor()
         
        while torch.sum(self.model._buffer_in._actual).item() != 0. or feed_count < inp_len:
            x_feed = x[:, feed_count, :] if feed_count < inp_len else torch.zeros(batch_size, self.params.input_size)
            z_tensor = torch.cat([z_tensor, 
                                #torch.zeros(self.params.batch_size, 1, 1),
                                torch.zeros(batch_size, 1, 1),
                                ],
                                1)
            z_tensor[:, feed_count, :] = self.model.z#[:, :]#.clone()
            
            #size of output: (batch_size, output_size)
            output = self.model(x_feed)
            outputs_tensor = torch.cat([outputs_tensor, 
                                       torch.zeros(batch_size, 1, self.output_size()),
                                       ],
                                        1)
            outputs_tensor[:, feed_count, :] = output[:, :].clone()
            feed_count += 1
        #print("Forward computation completed.")
        #size of outputs_tensor: (batch_size, characters, output_size)
        #size of z_tensor: (batch_size, characters, 1)
        #threshod = 0.99 threshold > 1 - 1/(inp)
        _, nonlambda_output_indices = torch.topk(z_tensor, inp_len, dim=1)
        nlo_indices = nonlambda_output_indices
        # bi: index in batch
        # ci: tensor(1): index of nlo in outputs_tensor
        
        for bi, sample in enumerate(nlo_indices):
            # loop for count the original y
            c_index = 0
            for ci in sample:
                ot_pred = outputs_tensor[bi, ci[0]].clone().view(1, -1)
                ot = y[bi, c_index].clone()
                if sum(x[bi, c_index]) != 0.:
                    batch_loss += self.loss_func(ot_pred, ot)
                    is_correct = 1 if torch.topk(ot_pred, 1)[1][0] == torch.topk(ot, 1)[1][0] else 0
                    batch_correct += is_correct
                    batch_total += 1
                c_index += 1
        # Regularization
        # pass
        #Update
        if is_batch:
            self.optimizer.zero_grad()
            batch_loss.backward()
            #batch_loss.backward(retain_graph=True)
            if self.clipping_norm:
                nn.utils.clip_grad_norm(self.model.parameters(),
                                        self.clipping_norm)
            self.optimizer.step()
        # Log the results.
        self._print_batch_summary(name, is_batch, batch_loss, batch_correct,
                                  batch_total)

        # Make the accuracy accessible for early stopping.
        self.batch_acc = batch_correct / batch_total
        consume_time = time.time() - start_time
        #print("Bacth %s: cosume %ds actual %d steps average %f s/step" % (name, consume_time, feed_count, consume_time/feed_count))
    def evaluate(self, epoch):
        if self.test_x is None or self.test_y is None:
            raise ValueError("Missing testing data")

        self.model.eval()

        # Embed the input data if necessary.
        if self.embedding is None:
            test_x = self.test_x
        else:
            test_x = self.embedding(self.test_x)
        self.model.init_model(test_x.size()[0])
        self._evaluate_batch(self.test_x, self.test_y, epoch, False)
        
    def _print_experiment_start(self):
        """
        Prints information about this Task's hyperparameters at the
        start of each experiment.
        """
        if not self.verbose:
            return

        print(("Starting {} Experiment".format(type(self).__name__)))
        self.model.print_experiment_start()
        self.params.print_experiment_start()

class PDACFGTask(PDATask):
    def input_size(self):
        return 6
    def output_size(self):
        return 2
    def get_data(self):
        D = {'s': [1, 0, 0, 0 ,0, 0],
            'e': [0, 0, 0, 0 ,0, 1],
            '[': [0, 1, 0, 0 ,0, 0],
            ']': [0, 0, 1, 0 ,0, 0],
            '{': [0, 0, 0, 1 ,0, 0],
            '}': [0, 0, 0, 0 ,1, 0],
            'l': [0, 0, 0, 0 ,0, 0],
            }
        DO = {'0': [0], 
              '1': [1],
              }
        trd_path = self.trd_path
        ted_path = self.ted_path
        trd = pd.read_csv(trd_path, header=None, dtype={0: str, 1: str})
        trd_x = trd.iloc[:, 0]
        trd_y = trd.iloc[:, 1]
        ted = pd.read_csv(ted_path, header=None, dtype={0: str, 1: str})
        ted_x = ted.iloc[:, 0]
        ted_y = ted.iloc[:, 1]
        
        trd_x1 = [list(s) for s in list(trd_x)]
        trd_y1 = [list(s) for s in list(trd_y)]
        ted_x1 = [list(s) for s in list(ted_x)]
        ted_y1 = [list(s) for s in list(ted_y)]
    
        for i, s in enumerate(trd_x1):
            for j, c in enumerate(s):
                trd_x1[i][j] = D[c]
        self.train_x = torch.Tensor(trd_x1)
        for i, s in enumerate(trd_y1):
            for j, c in enumerate(s):
                trd_y1[i][j] = DO[c]
        self.train_y = torch.Tensor(trd_y1).long()
        for i, s in enumerate(ted_x1):
            for j, c in enumerate(s):
                ted_x1[i][j] = D[c]
        self.test_x = torch.Tensor(ted_x1)
        for i, s in enumerate(ted_y1):
            for j, c in enumerate(s):
                ted_y1[i][j] = DO[c]
        self.test_y = torch.Tensor(ted_y1).long()
        