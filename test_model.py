'''

@author: lenovo
'''
from models.vanilla import PDAVanillaModel
from controllers.recurrent import PDALSTMSimpleStructController
from structs.simple import PDAStack
import torch
input_size = 8
read_size = 2
output_size = 1
batch_size = 1
controller_type = PDALSTMSimpleStructController
struct_type = PDAStack
model = PDAVanillaModel(input_size, read_size, output_size, controller_type, struct_type, )
model.init_model(batch_size)
inp = torch.FloatTensor(batch_size, input_size)

print(model(inp))
print(model(None))