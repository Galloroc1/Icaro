import numpy as np
from config import Config
from functools import reduce
from graph import build_graph


class BaseMatrix:

    def __init__(self, value, rank, ranks):
        self.value = value
        self.rank = rank
        self.ranks = ranks
        self.shape = value.shape

    def __add__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __mul__(self, other):
        return self.value * other

    def __divmod__(self, other):
        return self.value / other


class Tensor:
    def __init__(self, ops, ):
        self.compute_type = Config.compute_type
        self.ranks = []
        self.max_iter = 100000
        self.values = {}
        self.chunks = len(self.ranks)
        self.chunks_shapes = []
        self.ops = ops

    @property
    def shapes(self):
        if self.compute_type == 'single':
            return self.values[0].shape
        else:
            return reduce(lambda item1, item2: (item1.shape[0] + item2.shape[0], item1.shape[1] + item2.shape[1]),
                          self.values.values())

    def build_tensor(self, value):
        if Config.compute_type == 'single':
            self.values.update({0: value})

        if Config.compute_type == 'multiprocess':
            self.ranks = list(range(len(value)))
            self.chunks = len(self.ranks)
            for i in self.ranks:
                self.values.update({i: BaseMatrix(value=value[i], rank=i, ranks=self.ranks)})
                self.chunks_shapes.append(value[i].shape)

        return self

    def __add__(self, other):
        return Tensor(ops=[add, self, scaler(other)])

    def __sub__(self, other):
        return Tensor(ops=[sub, self, scaler(other)])

    def send(self):
        pass

    def recv(self):
        pass

    def compute(self):
        build_graph(self)
        # r = get_ops_result(*self.ops)
        # return reduce(lambda item1, item2: np.concatenate((item1, item2), axis=0), r)


def scaler(x):
    if isinstance(x, Tensor):
        return x
    else:
        return toTensorMulti(x)


def get_ops_result(op, tensor, y):
    if not tensor.ops:
        return op(tensor.values.values(), y)
    else:
        return op(get_ops_result(*tensor.ops), y)


def add(x, y):
    return x
    # return list(map(lambda item:item + y,x))


def sub(x, y):
    return x
    # return list(map(lambda item:item - y,x))


def toTensor(data, partitions=2):
    if Config.compute_type == 'single':
        pass


def toTensorSingle(data):
    if isinstance(data, np.ndarray):
        return data
    else:
        return Tensor(ops=[]).build_tensor(np.array(data))


def toTensorMulti(data, partitions=2):
    assert partitions > 0, f'you should make true partitions'
    if len(data) < partitions:
        partitions = len(data)
    return Tensor(ops=[]).build_tensor(np.array_split(np.array(data), partitions))


def toTensorCluster(data, partitions=2):
    pass


import datetime

Config.compute_type = 'multiprocess'
st = datetime.datetime.now()
data = np.random.random_sample((10, 2))
data_tensor = toTensorMulti(data)
data_tensor1 = data_tensor + 1.2

data2 = np.random.random_sample((10, 2))
data_tensor2 = toTensorMulti(data)

d3 = data_tensor2 + data_tensor1 + 4
print(data_tensor2.ops)
d4 = data_tensor2 + 1.1
d5 = d3 + d4
d5.compute()



