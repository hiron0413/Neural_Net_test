import numpy as np
from training import *

net = network()
train_data, test_data = generator(net, 10000, 1000)

for i in range(3):
    size = 14 - 2 * i
    np.save(f"data/train_data_{size}x{size}.npy", train_data[f"{size}x{size}"])
    np.save(f"data/test_data_{size}x{size}.npy", test_data[f"{size}x{size}"])

print("end")