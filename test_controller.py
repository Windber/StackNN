from controllers.recurrent import PDALSTMSimpleStructController
import torch 
if __name__ == "__main__":
    batch_size = 1
    input_size = 8
    read_size = 8
    output_size = 1
    custom_init = False
    discourage_pop = False
    hidden_size = 8
    n_args = 4
    controller = PDALSTMSimpleStructController(input_size, read_size, output_size, custom_init, discourage_pop, hidden_size, n_args,)
    controller.init_controller(batch_size)
    i = torch.FloatTensor(batch_size, input_size)
    r = torch.FloatTensor(batch_size, read_size)
    print(controller(i, r))