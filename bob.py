from config import Config
Config.init(role='host',
            name="bob",
            com_map={"alice": 7072, "bob": 8082})

from tensor.core import toTensor, array_to_tensor
import numpy as np

if __name__ == '__main__':

    data = toTensor(np.random.random_sample((8, 2)))
    y = toTensor(np.random.random_sample((8, 1)))
    w = toTensor(np.random.random_sample((2, 1)))
    b = toTensor(np.random.random_sample((1, 1)))
    pred = data.dot(w) + b

    alice_pred = toTensor(None, who="alice", dname='pred')

    grad = (alice_pred + pred)

    # grad.send("host", dname='grad')


