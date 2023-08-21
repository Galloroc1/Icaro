from tensor.tensor import toTensor,dot
import numpy as np
from task.config import *


if __name__ == '__main__':
    data = np.random.random_sample((3,4))
    en_data = toTensor(data)
    data2 = np.random.random_sample((4,2))
    en_data2 = toTensor(data2).encrypt()
    data3 = dot(en_data, en_data2).to_other()
    com.send()