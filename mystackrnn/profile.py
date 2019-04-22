from mystackrnn.stackrnntask import StackRNNTask
config_dyck2 = {
    "alphabet": {"(": [0], ")": [1], "s": [2], "e": [3], "#": [3], "[": [4], "]": [5]},
    "classes": {"0": 0, "1": 1},
    "batch_size": 100,
    "input_size": 6,
    "hidden_size": 8,
    "read_size": 8,
    "output_size": 2,
    "epochs": 1,
    "n_args": 3,
    "testfile_num": 1,
    "trpath": r"C:\Users\lenovo\git\rnn\data\Dyck2_dealed\dyck2_test1",#r"C:\Users\lenovo\git\rnn\data\Dyck2_dealed\dyck2_train",
    "tepath_prefix": r"C:\Users\lenovo\git\rnn\data\Dyck2_dealed\dyck2_test",
    "load": True,
    "saved_path_prefix": r"smodel\Time@",
    "load_path": r"smodel\Time@230031_0.59_0.71",
    "onlytest": False,
            }

if __name__ == "__main__":
    task = StackRNNTask(config_dyck2)
    task.experiment()