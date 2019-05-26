import numpy as np
from classifier import Classifier


class Svm(Classifier):
    def __init__(self, train_set, test_set, epochs=10, eta=0.1, lamda=0.1,
                 ratio=None, validate_set=None, num_classes=3):
        Classifier.__init__(self, train_set, test_set, epochs, eta, ratio, validate_set, num_classes)
        self.lamda = lamda

    def train(self):
        for e in range(self.epochs):
            for x, y in self.train_set:
                eta = self.eta / np.sqrt(e + 1)
                x = np.array(x)
                y = int(y)

                temp = np.dot(self.w, x)
                for t in range(len(temp)):
                    temp[t] = temp[t] + (y != t)
                y_hat = int(np.argmax(temp))

                loss = (y != y_hat) - np.dot(self.w[y], x) + np.dot(self.w[y_hat], x)
                self.w[y] = self.w[y] * (1 - eta * self.lamda)
                self.w[y_hat] = self.w[y_hat] * (1 - eta * self.lamda)
                if loss > 0:
                    self.w[y] = self.w[y] + eta * x
                    self.w[y_hat] = self.w[y_hat] - eta * x

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
            algo = Svm(train, set, self.epochs, self.eta, self.lamda)
            algo.train()
            acc += algo.validate(sets[i])
        return acc / k
