import numpy as np
from classifier import Classifier


class Perceptron(Classifier):
    def __init__(self, train_set, test_x, epochs=10, eta=0.1, ratio=None, validate_set=None, num_classes=3):
        Classifier.__init__(self, train_set, test_x, epochs, eta, ratio, validate_set, num_classes)

    def train(self):
        for e in range(self.epochs):
            for x, y in self.train_set:
                x = np.array(x)
                y = int(y)
                y_hat = int(np.argmax(np.dot(self.w, x)))
                if y != y_hat:
                    self.w[y] = self.w[y] + self.eta * x
                    self.w[y_hat] = self.w[y_hat] - self.eta * x

    # def train(self):
    #     for e in range(self.epochs):
    #         for x, y in self.train_set:
    #             x, y, E = np.array(x), int(y), []
    #             for i in range(len(self.w)):
    #                 if i != y and np.dot(self.w[i], x) > np.dot(self.w[y], x):
    #                     E.append(i)
    #             self.w[y] = self.w[y] + self.eta * x
    #             for j in E:
    #                 self.w[j] = self.w[j] - self.eta * (1 / len(E)) * x

    def k_cross_validation(self, k=6):
        sets = np.array_split(np.array(self.train_set), k)

        acc = 0
        for i in range(k):
            b = 1
            if i != 0:
                train = np.array(sets[0])
            else:
                b += 1
                train = np.array(sets[1])
            for j in range(b, k):
                if i != j:
                    train = np.concatenate((train, sets[j]))
            perceptron = Perceptron(train, set, self.epochs, self.eta)
            perceptron.train()
            acc += perceptron.validate(sets[i])
        return acc / k
