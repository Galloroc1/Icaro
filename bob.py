from config import Config
Config.init(role='host',
            name="bob",
            com_map={"alice": 7072, "bob": 8082})

from tensor.core import toTensor
import numpy as np

if __name__ == '__main__':

    data = toTensor(np.random.random_sample((8, 2)))
    y = toTensor(np.random.random_sample((8, 1)))
    alice_pred = toTensor(None, who="alice", dname='pred')
    grad = (y-alice_pred).mean()
    grad.send("host", dname='grad')


