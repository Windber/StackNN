"""
Because the problem on relative importing when launched as a individial script,
For convinience please launch ../test_simple_example.py which wraps simple_example.py 
Unit tests and usage examples for SimpleStructs.

"""
import torch
from torch.autograd import Variable


from .simple import PDAStack, PDAQueue, to_string
# comment out for not being used and nowhere to find testing module
#from testing import testcase


def test_push(s_struct, s1, s2, v1, v2):
    v_str = to_string(v1)
    s_str = to_string(s1.data)

    print("\nPushing {} with strength {}".format(v_str, s_str))
    v_str = to_string(v2)
    s_str = to_string(s2.data)

    print("\nPushing {} with strength {}".format(v_str, s_str))
    
    s_struct.push(s1, s2, v1, v2)
    s_struct.log()

    return


def test_pop(s_struct, strength):
    print("\nPopping with strength {:4f}".format(strength))
    s_struct.pop(strength)
    s_struct.log()

    return


def test_read(s_struct, strength):
    s_str = to_string(strength)

    print("\nReading with strength {}".format(s_str))
    print(to_string(s_struct.read(strength).data))

    return


test_stack = True  # Whether we are testing a Stack or a Queue
batch_size = 1  # The size of our mini-batches
embedding_size = 1  # The size of vectors held by the SimpleStruct

# Create a struct
if test_stack:
    struct = PDAStack(embedding_size)
else:
    struct = PDAQueue(embedding_size)

struct._init_Struct(batch_size)

# Push something
v1 = Variable(torch.randn(batch_size, embedding_size))
v2 = Variable(torch.randn(batch_size, embedding_size))
v3 = Variable(torch.randn(batch_size, embedding_size))
v4 = Variable(torch.randn(batch_size, embedding_size))
s1 = Variable(torch.FloatTensor([[1.]]))
s2 = Variable(torch.FloatTensor([[1.]])) if test_stack else Variable(torch.FloatTensor([[0.]]))

print(struct.actual)
test_push(struct, s1, s2, v1, v2)
test_push(struct, s1, s2, v3, v4)


# Read u then Pop u (actually what poped is just what read)
u = 0.4
print(struct.actual)
test_read(struct, u)
test_pop(struct, u)


u = 7.0
print(struct.actual)
test_read(struct, u)
test_pop(struct, u)


u = 1.4
print(struct.actual)
test_read(struct, u)
test_pop(struct, u)


