import numpy as np
import abc
import random


class Classifier:
    def __init__(self, train_set, test_x, epochs=100, eta=0.2, ratio=None, validate_set=None, num_classes=3):
        random.shuffle(train_set)
        self.train_set = train_set
        self.test_x = test_x
        self.validate_set = validate_set
        self.num_classes = num_classes
        self.ratio = ratio
        self.epochs = epochs
        self.eta = eta
        self.num_features = len(self.train_set[0][0])
        self.num_points = len(self.train_set)
        self.w = np.zeros((self.num_classes, self.num_features))
        # self.w = np.random.uniform(...))

        if self.ratio is not None:
            self.train_set = train_set[:int(self.num_points * self.ratio)]
            self.validate_set = train_set[int(self.num_points * self.ratio):]
            self.num_points = len(self.train_set)
        else:
            self.validate_set = validate_set

    def predict(self, X):
        y_hat = np.argmax(np.dot(self.w, X))
        return y_hat

    def predict_all(self):
        self.test_y = []
        for x in self.test_x:
            y = self.predict(x)
            self.test_y.append(y)

    @abc.abstractmethod
    def train(self):
        return

    @abc.abstractmethod
    def k_cross_validation(self, k=6):
        return

    def validate(self, validate_set=None):
        correct = 0
        if validate_set is None:
            validate_set = self.validate_set

        for x, y in validate_set:
            x = np.array(x)
            y_hat = np.argmax(np.dot(self.w, x))
            if y == y_hat:
                correct += 1

        incorrect = len(validate_set) - correct
        acc = (correct * 1.0) / ((correct + incorrect) * 1.0)
        return acc

    def k_validate(self, validate_set=None, k=6):
        if validate_set is None:
            validate_set = self.train_set
        if len(validate_set) < k:
            k = 1
        sets = np.array_split(np.array(validate_set), k)
        acc = 0
        for s in sets:
            acc += self.validate(s)
        return acc / k
