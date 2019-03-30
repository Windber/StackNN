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
from structs.simple import PDAStack
from .base import Task

class PDATask(Task, metaclass=ABCMeta):
    class Params:

        """Contains fully-specified parameters for this object.

        Parameters are either copied from kwargs are receive a default value.
        This inner class should be extended by subclasses of Task. The semantics
        of the parameter fields should be annotated in the class docstring.

        Attributes:
            model_type: A class extending Model.
            controller_type: A class extending SimpleStructController.
            struct_type: A class extending Struct.
            batch_size: The number of trials in each mini-batch.
            clipping_norm: Related to gradient clipping.
            criterion: The loss function.
            
            : If true and CUDA is available, the model will use it.
            epochs: Number of epochs to train for.
            early_stopping_steps: Number of epochs of no improvement that are
                required to stop early.
            hidden_size: The size of hidden state vectors.
            learning_rate: The learning rate.
            l2_weight: Float controlling the amount of L2 regularization.
            read_size: The length of vectors on the neural data structure.
            time_function: A function specifying the maximum number of
                computation steps in terms of input length.
            verbose: Boolean describing how much output should be generated.
            verbosity: Periodicity for printing batch summaries.
            load_path: Path for loading a model.
            save_path: Path for saving a model.
            test_override: Object describing params to override when evaluating
                a model for testing.
        """

        def __init__(self, **kwargs):
            """Extract passed arguments or use the default values."""
            self.model_type = kwargs.get("model_type", PDAVanillaModel)
            self.controller_type = kwargs.get(
                "controller_type", PDALSTMSimpleStructController)
            self.struct_type = kwargs.get("struct_type", Stack)
            self.batch_size = kwargs.get("batch_size", 10)
            self.clipping_norm = kwargs.get("clipping_norm", None)
            self.criterion = kwargs.get("criterion", nn.CrossEntropyLoss())
            self.cuda = kwargs.get("cuda", True)
            self.epochs = kwargs.get("epochs", 100)
            self.early_stopping_steps = kwargs.get("early_stopping_steps", 5)
            self.hidden_size = kwargs.get("hidden_size", 10)
            self.learning_rate = kwargs.get("learning_rate", 0.01)
            self.l2_weight = kwargs.get("l2_weight", 0.01)
            self.read_size = kwargs.get("read_size", 2)
            self.reg_weight = kwargs.get("reg_weight", 1.)
            self.time_function = kwargs.get("time_function", lambda t: t)
            self.verbose = kwargs.get("verbose", True)
            self.verbosity = kwargs.get("verbosity", 10)
            self.custom_initialization = kwargs.get("custom_initialization", True)
            self.load_path = kwargs.get("load_path", None)
            self.save_path = kwargs.get("save_path", None)
            self.test_override = kwargs.get("test_override", dict())


        def __iter__(self):
            return ((attr, getattr(self, attr)) for attr in dir(self)
                    if not attr.startswith("_"))

        def print_experiment_start(self):
            for key, value in self:
                if type(value) == type:
                    value = value.__name__
                print(("%s: %s" % (key, value)))

        @property
        def test(self):
            """ Get a Params object with test values set. """
            clone = deepcopy(self)
            for key, value in list(clone.test_override.items()):
                setattr(clone, key, value)

            return clone
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
        return self.params.model_type(self.input_size,
                                      self.params.read_size,
                                      self.output_size,
                                      controller_type=self.controller_type,
                                      struct_type=self.struct_type,
                                      hidden_size=self.hidden_size,
                                      reg_weight=self.reg_weight,
                                      custom_initialization=self.custom_initialization)
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
    @abstractmethod
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
        batch_loss = torch.zeros(1)
        batch_correct = 0
        batch_total = 0
        # size of actual_count: (batch_size, 1)
        #actual_count 
        feed_count = 0
        inp_len = x.size[1]
        outputs_tensor = torch.Tensor()
        z_tensor = torch.Tensor()
        # need to select another one for category task
        loss_fun = torch.nn.MSELoss(reduction='sum')
         
        while self.model._buffer_in._actual != 0. or feed_count < inp_len:
            x_feed = x[:, feed_count, :] if feed_count < inp_len else None
            z_tensor = torch.cat(z_tensor, 
                                torch.zeros(self.params.batch_size, 1, 1),
                                1)
            z_tensor[:, feed_count, :] = self.model.z[:, :].clone()
            
            #size of output: (batch_size, output_size)
            output = self.model(x_feed)
            outputs_tensor = torch.cat(outputs_tensor, 
                                       torch.zeros(self.params.batch_size, 1, self.output_size()),
                                       1)
            outputs_tensor[:, feed_count, :] = output[:, :].clone()
            feed_count += 1
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
                ot_pred = outputs_tensor[bi, ci[0]]
                ot = y[bi, c_index]
                batch_loss += loss_fun()
                c_index += 1
                is_correct = 1 if torch.topk(ot_pred, 1)[1][0] == torch.topk(ot, 1)[1][0] else 0
                batch_correct += is_correct
                batch_total += 1
        # Regularization
        # pass
        #Update
        if is_batch:
            self.optimizer.zero_grad()
            batch_loss.backward()
            if self.clipping_norm:
                nn.utils.clip_grad_norm(self.model.parameters(),
                                        self.clipping_norm)
            self.optimizer.step()
        # Log the results.
        self._print_batch_summary(name, is_batch, batch_loss, batch_correct,
                                  batch_total)

        # Make the accuracy accessible for early stopping.
        self.batch_acc = batch_correct / batch_total
    
    def evaluate(self, epoch):
        if self.test_x is None or self.test_y is None:
            raise ValueError("Missing testing data")

        self.model.eval()

        # Embed the input data if necessary.
        if self.embedding is None:
            test_x = self.test_x
        else:
            test_x = self.embedding(self.test_x)
        self.model.init_model(self.params.batch_size)
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