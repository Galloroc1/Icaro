from tensor.core import toTensor
import numpy as np

if __name__ == '__main__':

    data = toTensor(np.random.random_sample((8, 4)),)
    w = toTensor(np.random.random_sample((4, 1)))
    b = toTensor(np.random.random_sample((1, 1)))

    pred = (data.dot(w) + b)

    data2 = toTensor(np.random.random_sample((8, 1)))
    y = toTensor(np.random.random_sample((8, 1)))
    w2 = toTensor(np.random.random_sample((2, 1)))
    b2 = toTensor(np.random.random_sample((1, 1)))

    pred2 = (data2.dot(w) + b)

    grad = (pred + pred2)/y.shape[0]
    alpha = 0.001
    w = w - alpha * grad
    w2 = w2 - alpha * grad

