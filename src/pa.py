import numpy as np
from classifier import Classifier


class Pa(Classifier):
    def __init__(self, train_set, epochs=10, eta=None, ratio=None, validate_set=None, num_classes=3):
        Classifier.__init__(self, train_set, epochs, eta, ratio, validate_set, num_classes)

    def train(self):
        for e in range(self.epochs):
            for x, y in self.train_set:
                x = np.array(x)
                y = int(y)
                y_hat = int(np.argmax(np.dot(self.w, x)))

                loss = (y != y_hat) - np.dot(self.w[y], x) + np.dot(self.w[y_hat], x)
                if loss > 0:
                    tau = loss / (np.power(np.linalg.norm(x, ord=2), 2) * 2)
                    self.w[y] = self.w[y] + tau * x
                    self.w[y_hat] = self.w[y_hat] - tau * x

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
            algo = Pa(train, set, self.epochs)
            algo.train()
            acc += algo.validate(sets[i])
        return acc / k