from paillier.paillier import generate_paillier_keypair
import numpy as np
p,q = generate_paillier_keypair()
data = np.random.random_sample((2,2))
data2 = p.encrypt(data)
data = data2.dot(data)
print(data.exponent)