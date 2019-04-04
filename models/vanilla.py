


import torch
from torch.autograd import Variable

from .base import Model
from controllers.feedforward import LinearSimpleStructController
from stacknn_utils.errors import unused_init_param
from structs.simple import Stack, PDAStack, PDAQueue
from structs.base import PDAStruct
from stacknn_utils.profile import timeprofile

class PDAVanillaModel(Model):

    def __init__(self, input_size, read_size, output_size,
                 controller_type=LinearSimpleStructController, struct_type=PDAStack,
                 **kwargs):
        super(PDAVanillaModel, self).__init__(read_size, struct_type)
        
        self._controller_type = controller_type
        self._input_size = input_size
        self._output_size = output_size
        
        self._read = None
        self._z = None

        self._u = None
        self._s1 = None
        self._s2 = None
        self._v1 = None
        self._v2 = None
        
        self._controller = self._controller_type(input_size, read_size, output_size, **kwargs)
        self._struct = self._struct_type(self._read_size)
        self._buffer_in = PDAQueue(self._input_size)

        self._t = 0
        self._zeros = None

        
    
    def _init_struct(self, batch_size):
            #self._struct._read = Variable(torch.zeros([batch_size, self._read_size]))
            self._struct._init_Struct(batch_size)
            #self._read = Variable(torch.zeros(batch_size, self._read_size)
           
            
    def _init_buffer(self, batch_size, xs=None):
        # have input queue
        # dont have output queue
        self._buffer_in._init_Struct(batch_size)
        #self._readinput = Variable(torch.zeros(batch_size, self._input_size))
        if xs:
            pass
        #self._buffer_out = []

        self._t = 0
        self._zeros = Variable(torch.zeros(batch_size, self._input_size))

    """ Neural Network Computation """
    def init_model(self, batch_size):
        #super.init_model(batch_size, xs)
        self._init_struct(batch_size)
        self._init_buffer(batch_size)
        self._init_controller(batch_size)
        self.z = Variable(torch.ones(batch_size, 1))
        self.read = Variable(torch.zeros(batch_size, self._read_size))
        self._reg_loss = torch.zeros([batch_size, self._read_size])
        self.batch_size = batch_size
    def _init_controller(self, batch_size):
        self._controller.init_controller(batch_size)
        #self._z = self._controller.z
    @timeprofile
    def forward(self, inp=None):
        if self.read is None:
            raise RuntimeError("The data structure has not been initialized.")
        if self.z is None:
            raise RuntimeError("The data structure has not been initialized.")
        '''
        if inp:
            notl = True
        else:
            notl = sum(inp) != 0.
        input_strength = Variable(torch.ones(self.batch_size, 1)) if notl else Variable(torch.zeros(self.batch_size, 1))
        inp_pad = inp if notl else Variable(torch.zeros(self.batch_size, self._input_size))
        '''
        input_strength = torch.sum(inp, dim=1).view(-1, 1)
        
        inp_real = self._buffer_in(self.z, input_strength, Variable(torch.zeros(self.batch_size, 1)), inp, Variable(torch.zeros(self.batch_size, self._input_size))) 
        # output, v1, v2, (s1, s2, u, z)
        o, self._v1, self._v2, (self._s1, self._s2, self._u, self._z)= self._controller(inp_real, self.read)
        self.read = self._struct(self._u, self._s1, self._s2, self._v1, self._v2)
        return o

    """ Accessors """

    def _read_input(self):
        """
        Returns the next vector from the input buffer.

        :rtype: Variable
        :return: The next vector from the input buffer
        """
        if self._t < self._buffer_in.size(1):
            self._t += 1
            return self._buffer_in[:, self._t - 1, :]
        else:
            return self._zeros

    def read_output(self):
        """
        Returns the next vector from the output buffer.

        :rtype: Variable
        :return: The next vector from the output buffer
        """
        if len(self._buffer_out) > 0:
            return self._buffer_out.pop(0)
        else:
            return None

    def _write_output(self, value):
        """
        Adds a symbol to the output buffer.

        :type value: Variable
        :param value: The value to add to the output buffer

        :return: None
        """
        self._buffer_out.append(value)

    @property
    def read(self):
        return self._read
    @read.setter
    def read(self, r):
        self._read = r
    @property
    def z(self):
        return self._z
    @z.setter
    def z(self, z):
        self._z = z


class VanillaModel(Model):
    """
    A simple Model that uses a SimpleStruct as its data structure.
    """

    def __init__(self, input_size, read_size, output_size,
                 controller_type=LinearSimpleStructController, struct_type=Stack,
                 **kwargs):
        """
        Constructor for the VanillaModel object.

        :type input_size: int
        :param input_size: The size of the vectors that will be input to
            this Model

        :type read_size: int
        :param read_size: The size of the vectors that will be placed on
            the neural data structure

        :type output_size: int
        :param output_size: The size of the vectors that will be output
            from this Model

        :type struct_type: type
        :param struct_type: The type of neural data structure that this
            Model will operate

        :type controller_type: type
        :param controller_type: The type of the Controller that will perform
            the neural network computations
        """
        super(VanillaModel, self).__init__(read_size, struct_type)
        self._read = None
        self._controller = controller_type(input_size, read_size, output_size,
                                     **kwargs)
        self._input_size = input_size
        self._output_size = output_size
        self._read_size = read_size

        self._buffer_in = None
        self._buffer_out = None

        self._t = 0
        self._zeros = None

        self._push_input = kwargs.get("push_input", False)

    def _init_buffer(self, batch_size, xs):
        """
        Initializes the input and output buffers. The input buffer will
        contain a specified collection of values. The output buffer will
        be empty.

        :type batch_size: int
        :param batch_size: The number of trials in each mini-batch where
            this Model is used

        :type xs: Variable
        :param xs: An array of values that will be placed on the input
            buffer. The dimensions should be [batch size, t, read size],
            where t is the maximum length of a string represented in xs

        :return: None
        """
        self._buffer_in = xs
        self._buffer_out = []

        self._t = 0
        self._zeros = Variable(torch.zeros(batch_size, self._input_size))

    """ Neural Network Computation """

    def forward(self):
        """
        Computes the output of the neural network given an input. The
        controller should push a value onto the neural data structure and
        pop one or more values from the neural data structure, and
        produce an output based on this information and recurrent state
        if available.

        :return: None
        """
        if self._read is None:
            raise RuntimeError("The data structure has not been initialized.")

        x = self._read_input()

        output, (v, u, d) = self._controller(x, self._read)
        if self._push_input:
            v = x
        self._read = self._struct(v, u, d)

        self._write_output(output)

    """ Accessors """

    def _read_input(self):
        """
        Returns the next vector from the input buffer.

        :rtype: Variable
        :return: The next vector from the input buffer
        """
        if self._t < self._buffer_in.size(1):
            self._t += 1
            return self._buffer_in[:, self._t - 1, :]
        else:
            return self._zeros

    def read_output(self):
        """
        Returns the next vector from the output buffer.

        :rtype: Variable
        :return: The next vector from the output buffer
        """
        if len(self._buffer_out) > 0:
            return self._buffer_out.pop(0)
        else:
            return None

    def _write_output(self, value):
        """
        Adds a symbol to the output buffer.

        :type value: Variable
        :param value: The value to add to the output buffer

        :return: None
        """
        self._buffer_out.append(value)

    """ Reporting """

    def trace(self, trace_x, *args):
        """
        Draws a graphic representation of the neural data structure
        instructions produced by the Model's Controller at each time
        step for a single input.

        :type trace_x: Variable
        :param trace_x: An input string

        :return: None
        """
        for arg in args:
            unused_init_param("num_steps", arg, self)

        self.eval()
        self.init_model(1, trace_x)

        max_length = trace_x.data.shape[1]

        self._controller.start_log(max_length)
        for j in range(max_length):
            self.forward()
        self._controller.stop_log()

        x_labels = ["x_" + str(i) for i in range(self._input_size)]
        y_labels = ["y_" + str(i) for i in range(self._output_size)]
        i_labels = ["Pop", "Push"]
        v_labels = ["v_" + str(i) for i in range(self._read_size)]
        labels = x_labels + y_labels + i_labels + v_labels

        import matplotlib.pyplot as plt
        plt.imshow(self._controller.log_data, cmap="hot", interpolation="nearest")
        plt.title("Trace")
        plt.yticks(list(range(len(labels))), labels)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()

    def trace_step(self, trace_x, num_steps=None, step=True):
        """
        Steps through the neural network's computation. The controller will
        read an input and produce an output. At each time step, a
        summary of the controller's state and actions will be printed to
        the console.

        :type trace_x: Variable
        :param trace_x: A single input string

        :type num_steps: int
        :param num_steps: Please do not pass anything to this param

        :type step: bool
        :param step: If True, the user will need to press Enter in the
            console after each computation step

        :return: None
        """
        if num_steps is not None:
            unused_init_param("num_steps", num_steps, self)
        if trace_x.data.shape[0] != 1:
            raise ValueError("You can only trace one input at a time!")

        self.eval()
        self.init_model(1, trace_x)

        x_end = self._input_size
        y_end = x_end + self._output_size
        push = y_end + 1
        v_start = push + 1

        max_length = trace_x.data.shape[1]
        self._controller.start_log(max_length)
        for j in range(max_length):
            print("\n-- Step {} of {} --".format(j, max_length))

            self()

            i = self._controller.log_data[:x_end, j]
            o = self._controller.log_data[x_end:y_end, j].round(decimals=4)
            u = self._controller.log_data[y_end, j].round(decimals=4)
            d = self._controller.log_data[push, j].round(decimals=4)
            v = self._controller.log_data[v_start:, j].round(decimals=4)
            r = self._struct.read(1).data.numpy()[0].round(decimals=4)

            print("\nInput: " + str(i))
            print("Output: " + str(o))

            print("\nPop Strength: " + str(u))

            print("\nPush Vector: " + str(v))
            print("Push Strength: " + str(d))

            print("\nRead Vector: " + str(r))
            print("Struct Contents: ")
            self._struct.print_summary(0)

            if step:
                input("\nPress Enter to continue\n")
        self._controller.stop_log()
