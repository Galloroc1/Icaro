from tensor.tensor import toTensor,dot
import numpy as np
from config import Config
if __name__ == '__main__':
    Config.init(role='host',
                name="alice",
                com_map={"alice": 7072, "bob": 8082})
    data = np.random.random_sample((3,4))
    en_data = toTensor(data)
    data2 = np.random.random_sample((4,2))
    en_data2 = toTensor(data2).encrypt()
    data3 = dot(en_data, en_data2)
    print(data3.decrypt().compute())
    print(data.dot(data2))