'''
@author: lenovo
'''
from tasks.PDATask import PDACFGTask
from models.vanilla import PDAVanillaModel
from structs.simple import PDAStack
from controllers.recurrent import PDALSTMSimpleStructController

config_dict = dict()
config_dict['model_type'] = PDAVanillaModel
config_dict['controller_type'] = PDALSTMSimpleStructController
config_dict['struct_type'] = PDAStack
config_dict['batch_size'] = 1
config_dict['clipping_norm'] = None
config_dict['lifting_norm'] = None
config_dict['cuda'] = False
config_dict['epochs'] = 10
config_dict['hidden_size'] = 4
config_dict['learning_rate'] = 0.01
config_dict['read_size'] = 2
config_dict['task'] = PDACFGTask
config_dict['input_size'] = 6
config_dict['output_size'] = 2
config_dict['leafting_norm'] = 0.2
config_dict['custom_initialization'] = False
config_dict['trd_path'] = r'C:\Users\lenovo\Documents\Documents\kaiti\datasets\dyck2_train'
config_dict['ted_path'] = r'C:\Users\lenovo\Documents\Documents\kaiti\datasets\dyck2_test'
pct = PDACFGTask.from_config_dict(config_dict)
print(pct.run_experiment())