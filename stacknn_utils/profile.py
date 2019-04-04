'''
@author: lenovo
'''
import time
def timeprofile(f):
    def computetime(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        con = time.time() - start
        if con > 0.5:
            print("function %s comsumed %ds" % (f.__name__, con))
        return result
    return computetime