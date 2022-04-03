import numpy as np
import matplotlib.pyplot as plt
from training import *

net = network(10, 0.0005, False)
for k in range(2):
    train_data, test_data = load_data(3000, 200)

    train_loss = [0,0,0]
    test_loss = [0,0,0]
    train_accuracy = [0,0,0]
    test_accuracy = [0,0,0]

    train_accuracy[0], test_accuracy[0], train_loss[0], test_loss[0], n = train(net, 14, train_data, test_data, 500, 100, 0.03, 5, True)
    train_accuracy[1], test_accuracy[1], train_loss[1], test_loss[1], n = train(net, 12, train_data, test_data, 500, 100, 0.03, 5, True)
    train_accuracy[2], test_accuracy[2], train_loss[2], test_loss[2], n = train(net, 10, train_data, test_data, 500, 100, 0.03, 5, True)

    x = np.arange(len(test_loss[0]))

    #plt.ylim(24.0, 25.0)

    for i in range(2):
        for j in range(3):
            plt.subplot(2, 6, 6*k + 3*i + j + 1)
            plt.title("{size}x{size}".format(size = 14 - 2 * j))
            if i == 1:
                p1 = plt.plot(x, train_accuracy[j], linestyle="solid")
                p2 = plt.plot(x, test_accuracy[j], linestyle="dashed")
            else:
                p1 = plt.plot(x, train_loss[j], linestyle="solid")
                p2 = plt.plot(x, test_loss[j], linestyle="dashed")
            plt.legend((p1[0], p2[0]), ("train", "test"), loc=2)

    net = network(10, 0.0005, True)

plt.show()

"""
for i in range(3):
    size = 14 - 2 * i
    for key in ("W1", "b1", "W2", "b2"):
        np.savetxt(f"layers/{size}x{size}_{key}.csv", net[f"{size}x{size}"].params[key].flatten(), delimiter=',')
"""

"""
net = network(10, 0.5)
train_data, test_data = load_data(1000, 200)

optimization_trial = 100
results_test = {}
results_train = {}

for i in range(optimization_trial):
    weight_decay = 10 ** np.random.uniform(-7, -3)
    lr = 10 ** np.random.uniform(-3, -1)
    hidden = int(10 ** np.random.uniform(1, 2))
    net = network(hidden, weight_decay)

    _, train_loss, test_loss, n = train(net, 14, train_data, test_data, 200, 100, lr, 20)

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
"""