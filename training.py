import numpy as np
from sklearn.metrics import accuracy_score
from neural_net_1 import NeuralNet1

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

def network(hidden_size = 100, weight_decay_lambda = 0, use_sigmoid = True):
    net_14x14 = NeuralNet1(14, 7, (3,2,2,2,1), hidden_size, weight_decay_lambda, use_sigmoid)
    net_12x12 = NeuralNet1(12, 6, (3,2,2,1,0), hidden_size, weight_decay_lambda, use_sigmoid)
    net_10x10 = NeuralNet1(10, 5, (2,2,1,1,0), hidden_size, weight_decay_lambda, use_sigmoid)
    return {"14x14": net_14x14, "12x12": net_12x12, "10x10": net_10x10}

def generator(net, train = 10000, test = 200):
    train_data_14x14 = generate_data(net["14x14"], train, 14, 7)
    train_data_12x12 = generate_data(net["12x12"], train, 12, 6)
    train_data_10x10 = generate_data(net["10x10"], train, 10, 5)

    test_data_14x14 = generate_data(net["14x14"], test, 14, 7)
    test_data_12x12 = generate_data(net["12x12"], test, 12, 6)
    test_data_10x10 = generate_data(net["10x10"], test, 10, 5)
    return {"14x14": train_data_14x14, "12x12": train_data_12x12, "10x10": train_data_10x10}, {"14x14": test_data_14x14, "12x12": test_data_12x12, "10x10": test_data_10x10}

def load_data(_train = 10000, _test = 1000):
    train = min(_train, 10000)
    test = min(_test, 1000)
    train_data = []
    test_data = []
    for i in range(3):
        train_data.append(np.load("data/train_data_{size}x{size}.npy".format(size = 14 - 2 * i), allow_pickle=True))
        test_data.append(np.load("data/test_data_{size}x{size}.npy".format(size = 14 - 2 * i), allow_pickle=True))
    return {"14x14": train_data[0][:train], "12x12": train_data[1][:train], "10x10": train_data[2][:train]}, {"14x14": test_data[0][:test], "12x12": test_data[1][:test], "10x10": test_data[2][:test]}

def train(net, size, train_data, test_data, iter_num = 10000, _batch_size = 100, learning_rate = 0.01, _iter_per_epoch = 20, get_accuracy = False):
    if not size in (10, 12, 14):
        raise ValueError("Size must be 10, 12, or 14.")
    train_size = train_data[f"{size}x{size}"][1].shape[0]
    batch_size = min(_batch_size, int(train_size / 20))
    network = net[f"{size}x{size}"]
    train_loss = []
    train_loss2 = []
    test_loss = []
    iter_per_epoch = min(_iter_per_epoch, int(train_size / 20))
    n = 0
    train_accuracy = []
    test_accuracy = []

    for i in range(iter_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = train_data[f"{size}x{size}"][1][batch_mask]
        t_batch = train_data[f"{size}x{size}"][0][batch_mask]

        loss, gradients = network.backpropagation(x_batch, t_batch)

        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * gradients[key]

        if i % iter_per_epoch == 0:
            loss = network.loss(train_data[f"{size}x{size}"][1], train_data[f"{size}x{size}"][0])
            train_loss.append(loss)
            if get_accuracy:
                accuracy = network.accuracy(train_data[f"{size}x{size}"][1], train_data[f"{size}x{size}"][0])
                train_accuracy.append(accuracy)
            #print(loss)
            loss = network.loss(test_data[f"{size}x{size}"][1], test_data[f"{size}x{size}"][0])
            test_loss.append(loss)
            if get_accuracy:
                accuracy = network.accuracy(test_data[f"{size}x{size}"][1], test_data[f"{size}x{size}"][0])
                test_accuracy.append(accuracy)
            #print(loss)
            n += 1

    return train_accuracy, test_accuracy, train_loss, test_loss, n