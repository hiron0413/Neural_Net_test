from typing import OrderedDict
import numpy as np
import math
import random

from Layers import *

class NeuralNet1:
    def __init__(self, w, h, ships: tuple, hidden_size = 100, weight_decay_lambda = 0):
        self.width = w
        self.height = h
        self.ships = ships
        self.weight_decay_lambda = weight_decay_lambda
        self.tiles = [[0 for _ in range(w)] for __ in range(h)]
        self.dirs = [1,-1]
        self.input = None

        self.params = {}
        self.params["W1"] = np.random.randn(w*h, hidden_size) / np.sqrt(w*h)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = np.random.randn(hidden_size, w*h) / np.sqrt(hidden_size)
        self.params["b2"] = np.zeros(w*h)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["ReLU1"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.lastLayer = IdentityWithLoss()
        
    def generate_data(self):
        self.tiles = [[0 for _ in range(self.width)] for __ in range(self.height)]
        r = random.random()
        for i in range(5, 0, -1):
            for j in range(self.ships[i-1]):
                result = False
                while not result:
                    result = self.add_ship(random.randint(0, self.width - 1), random.randint(0, self.height - 1), self.dirs[random.randint(0, 1)], [True, False][random.randint(0, 1)], i)
        self.input = np.array([0 if random.random() < r else self.tiles[j][i] + 1 for i in range(self.width) for j in range(self.height)])
        tiles = np.array([self.tiles[j][i] for i in range(self.width) for j in range(self.height)])
        return [tiles, self.input]
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        out = self.predict(x)

        weight_decay = 0
        weight_decay += 0.5 * self.weight_decay_lambda * np.sum(self.params["W1"] ** 2)
        weight_decay += 0.5 * self.weight_decay_lambda * np.sum(self.params["W2"] ** 2)
        
        return self.lastLayer.forward(out, t) + weight_decay

    def backpropagation(self, x, t):
        # ->
        loss = self.loss(x, t)

        # <-
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        gradients = {}

        gradients["W1"] = self.layers["Affine1"].dW + self.weight_decay_lambda * self.layers["Affine1"].W
        gradients["b1"] = self.layers["Affine1"].db
        gradients["W2"] = self.layers["Affine2"].dW + self.weight_decay_lambda * self.layers["Affine2"].W
        gradients["b2"] = self.layers["Affine2"].db

        return loss, gradients


    def add_ship(self, x, y, dir, vertical, l):
        if (not vertical and (x + dir * l > self.width - 1 or x + dir * l < 0)) or (vertical and (y + dir * l > self.height - 1 or y + dir * l < 0)):
            return False

        list = []
        if vertical:
            for i in range(1, l + 1):
                list.append([x, y + i * dir])
                if self.tiles[y + i * dir][x] == 1: return False
        else:
            for i in range(1, l + 1):
                list.append([x + i * dir, y])
                if self.tiles[y][x + i * dir] == 1: return False
        
        for pos in list:
            self.tiles[pos[1]][pos[0]] = 1
        return True