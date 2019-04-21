
'''
@author: lenovo
'''
from tasks.PDATask import PDACFGTask
from models.vanilla import PDAVanillaModel
from structs.simple import PDAStack
from controllers.recurrent import PDALSTMSimpleStructController, PDAGRUSimpleStructController, PDARNNSimpleStructController
import time

'''
train 800 20(6)
validation 100 
test 1000 110(12)
epoch 10
h 10
read_size 2
batch_size 10

'''
config_dict = dict()
config_dict['model_type'] = PDAVanillaModel
config_dict['controller_type'] = PDAGRUSimpleStructController
config_dict['struct_type'] = PDAStack
config_dict['batch_size'] = 100
config_dict['clipping_norm'] = 10
config_dict['leafting_norm'] = 0.02
config_dict['lifting_norm'] = None
config_dict['cuda'] = False
config_dict['epochs'] = 100
config_dict['hidden_size'] = 8
config_dict['learning_rate'] = 1e-3
config_dict['read_size'] = 2
config_dict['task'] = PDACFGTask
config_dict['input_size'] = 6
config_dict['output_size'] = 2
config_dict['custom_initialization'] = False
config_dict['trd_path'] = r'./data/train_32_8_16000'
config_dict['ted_path'] = r'./data/test_32_8_1000'
config_dict['save_path'] = r'./savedmodel/the best_RNN_model'
config_dict['load_path'] = r'./savedmodel/the best_RNN_model@21_01_46'
config_dict["load"] = False
config_dict['cross_validation'] = False
config_dict['kfold'] = 10
config_dict['model'] = "manytomany"
config_dict['clampstep'] = 64
pct = PDACFGTask.from_config_dict(config_dict)
if not pct._has_trained_model:
    acc = pct.run_experiment()
    print(acc)
else:
    pct.get_data() 
pct.test()
print(pct.probe)
