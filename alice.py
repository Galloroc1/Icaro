from tensor.tensor import toTensor
import numpy as np
from communication.core import Communicate


if __name__ == '__main__':
    com = Communicate(role='host',
                      name="alice",
                      port=9082,
                      other='bob',
                      other_port=9072)

    data = np.random.random_sample((2,2))
    en_data = toTensor(data)
    data2 = np.random.random_sample((2,2))
    en_data2 = toTensor(data2).encrypt()
    data3 = en_data2.dot(en_data)
    data3.to_other(com, dname="dot")
    print(data3.dot(toTensor(np.ones(shape=(data3.shape[1:])))).decrypt().compute())
    print(data2.dot(data).dot(np.ones(shape=(data3.shape[1:]))))
    # data4 = toTensor(None,com,dname="dot2")
    # de_data = data4.encrypt()
    # print("decrypt\n", de_data)
    # print("org\n",data2.dot(data).dot(np.zeros(shape=(data3.shape[1:]))))
