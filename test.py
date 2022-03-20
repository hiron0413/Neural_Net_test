import sys
import os
from typing import Dict
import numpy as np
from neural_net_1 import NeuralNet1
import matplotlib.pyplot as plt

net = None
train_data = None
test_data = None

def generate_data(network, n = 500, w=14, h=7):
    global x
    data = network.generate_data()
    x = [data[1]]
    t = [data[0]]

    for i in range(n-1):
        data = network.generate_data()
        x = np.append(x, [data[1]], axis=0)
        t = np.append(t, [data[0]], axis=0)
    
    return np.array([x,t])

def network(hidden_size = 100, weight_decay_lambda = 0):
    net_14x14 = NeuralNet1(14, 7, (3,2,2,2,1), hidden_size, weight_decay_lambda)
    net_12x12 = NeuralNet1(12, 6, (3,2,2,2,1), hidden_size, weight_decay_lambda)
    net_10x10 = NeuralNet1(10, 5, (3,2,2,2,1), hidden_size, weight_decay_lambda)
    return {"14x14": net_14x14, "12x12": net_12x12, "10x10": net_10x10}

def generator(net, train = 10000, test = 200):
    train_data_14x14 = generate_data(net["14x14"], train, 14, 7)
    train_data_12x12 = generate_data(net["12x12"], train, 12, 6)
    train_data_10x10 = generate_data(net["10x10"], train, 10, 5)

    test_data_14x14 = generate_data(net["14x14"], test, 14, 7)
    test_data_12x12 = generate_data(net["12x12"], test, 12, 6)
    test_data_10x10 = generate_data(net["10x10"], test, 10, 5)
    return {"14x14": train_data_14x14, "12x12": train_data_12x12, "10x10": train_data_10x10}, {"14x14": test_data_14x14, "12x12": test_data_12x12, "10x10": test_data_10x10}


def train(size, train_data: Dict, test_data: Dict, iter_num = 10000, _batch_size = 100, learning_rate = 0.01, _iter_per_epoch = 20):
    if not size in (10, 12, 14):
        return False
    train_size = train_data[f"{size}x{size}"][1].shape[0]
    batch_size = min(_batch_size, int(train_size / 20))
    network = net[f"{size}x{size}"]
    train_loss = []
    train_loss2 = []
    test_loss = []
    iter_per_epoch = min(_iter_per_epoch, int(train_size / 20))
    n = 0

    for i in range(iter_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = train_data[f"{size}x{size}"][1][batch_mask]
        t_batch = train_data[f"{size}x{size}"][0][batch_mask]

        loss, gradients = network.backpropagation(x_batch, t_batch)

        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * gradients[key]
        
        train_loss.append(loss)

        if i % iter_per_epoch == 0:
            loss = network.loss(train_data[f"{size}x{size}"][1], train_data[f"{size}x{size}"][0])
            train_loss2.append(loss)
            #print(loss)
            loss = network.loss(test_data[f"{size}x{size}"][1], test_data[f"{size}x{size}"][0])
            test_loss.append(loss)
            #print(loss)
            n += 1

    return train_loss, train_loss2, test_loss, n

net = network(100, 0.5)
train_data, test_data = generator(net, 1000, 200)

optimization_trial = 200
results_test = {}
results_train = {}

for i in range(optimization_trial):
    weight_decay = 10 ** np.random.uniform(-7, -3)
    lr = 10 ** np.random.uniform(-3, -1)
    hidden = int(10 ** np.random.uniform(1, 2))
    net = network(hidden, weight_decay)

    _, train_loss, test_loss, n = train(14, train_data, test_data, 300, 100, lr, 20)

    key = "hidden:" + str(hidden) + ", lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_test[key] = test_loss
    results_train[key] = train_loss
    print(i / optimization_trial * 100)


graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, _ in sorted(results_test.items(), key=lambda x:x[1][-1], reverse=False):
    print("Best-" + str(i+1) + "(test loss:" + str(results_test[key][-1]) + ") | " + key)

    x = np.arange(len(results_train[key]))

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(22.0, 30.0)

    if i % col_num: plt.yticks([])
    plt.xticks([])

    plt.plot(x, results_train[key], linestyle="solid")
    plt.plot(x, results_test[key], linestyle="dashed")
    #plt.legend((p1[0], p2[0]), ("train", "test"), loc=2)

    i += 1

    if i >= graph_draw_num:
        break

plt.show()