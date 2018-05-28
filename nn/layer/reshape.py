import numpy as np


def reshape(shape):
    global a
    a = np.reshape(a, shape)
    print a

a = np.arange(0, 24)
print a
reshape([2, 3, 4, 1])
a = np.transpose(a, [1, 2, 0, 3])
print a
reshape([-1, 2, 1])
