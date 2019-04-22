from models.base import Model
class StackRNNCell(Model):

    def __init__(self, batch_size, input_size, hidden_size, read_size, output_size,
                 controller_type, struct_type,
                 **kwargs):
        super().__init__(read_size, struct_type)
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.read_size = read_size
        self.batch_size = batch_size
        
        
        self._read = None
        self._u = None
        self._s1 = None
        self._s2 = None
        self._v1 = None
        self._v2 = None
        #self._z = None
        #self._zo = None

        
        self._ol = None
        
        self.controller = controller_type(input_size, hidden_size, read_size, output_size, **kwargs)
        self._struct = self._struct_type(self._read_size)
        self._buffer_in = PDAQueue(self._input_size)
        self._buffer_out = PDAQueue(self._output_size)
        self._t = 0
        self._zeros = None

        
    
    def _init_struct(self, batch_size):
            #self._struct._read = Variable(torch.zeros([batch_size, self._read_size]))
            self._struct._init_Struct(batch_size)
            #self._read = Variable(torch.zeros(batch_size, self._read_size)
           
            
    def _init_buffer(self, batch_size, xs=None):
        self._buffer_in._init_Struct(batch_size)
        self._buffer_out._init_Struct(batch_size)
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
    def forward(self, inp=None):
        
        #input_strength = torch.sum(inp, dim=1).view(-1, 1)
        #ibuflen_tminus1 = self._buffer_in._actual
        #inp_real = self._buffer_in(self.z, input_strength, torch.zeros(self.batch_size, 1), inp, Variable(torch.zeros(self.batch_size, self._input_size))) 
        #ibuflen_t = self._buffer_in._actual
        #self.ol = input_strength + ibuflen_tminus1 - ibuflen_t
        # output, v1, v2, (s1, s2, u, z)
        inp_real = inp
        #o, self._v1, self._v2, (self._s1, self._z)= self._controller(inp_real, self.read)
        o, self._v1, self._v2, (self._s1, self._s2, self._u)= self._controller(inp_real, self.read)
        #self._s2 = self._s1.clone()
        #self._zo = self._z.clone()
        #self._u = torch.ones(self.batch_size, 1)
        self.read = self._struct(self._u, self._s1, self._s2, self._v1, self._v2)
        #self._buffer_out(torch.zeros(self.batch_size, 1), self.ol, torch.zeros(self.batch_size, 1), o, torch.zeros(self.batch_size, 1))
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

