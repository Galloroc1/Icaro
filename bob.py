import numpy as np

from tensor.tensor import toTensor
from communication.core import Communicate


if __name__ == '__main__':
    com = Communicate(role='guest',
                      name="bob",
                      port=8072,
                      other='alice',
                      other_port=8082)

    en_data = toTensor(None,com,dname='dot')
    other = toTensor(np.ones(shape=en_data.shape[1:]))
    data = en_data.dot(other)
    data.to_other(com, dname="dot2")
