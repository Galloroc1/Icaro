import dask
from config import Config
Config.init(role='host',
            name="alice",
            com_map={"alice": 7072, "bob": 8082})

from tensor.paillier_single import generate_paillier_keypair
from tensor.core import toTensor, array_to_tensor
import numpy as np

if __name__ == '__main__':
    # # local process
    # dask scheduler
    # client = Client(n_workers=12, threads_per_worker=2)
    # # dask worker 127.0.0.1:8786 --nworkers 8 --nthreads 2
    from dask.distributed import Client

    client = Client(address='127.0.0.1:8786')

    data = toTensor(np.random.random_sample((8, 4)))
    w = toTensor(np.random.random_sample((4, 1)))
    b = toTensor(np.random.random_sample((1, 1)))

    pred = (data.dot(w) + b).encrypt()
    pred.send("bob", dname="pred")

    grad = toTensor(None, who="bob", dname='grad')
    alpha = 0.001
    w = w - w * grad

    # grad = toTensor(None)
    # grad.get("bob", "grad")
    #
    # # alice have self's private_key and public_key
    # grad = toTensor(None, belong="guest", dname='grad').decrypt()

