# Steps
# Training:
# -> Initialise weights as zero
# -> Initialise bias as zero
# Given a data point:
# -> predict result by using y' = 1 / (1 + e^(-wx+b))
# -> calculate error
# -> use gradient descent to figure out new weight and bias values
# -> repeat n times

# Testing
# Given a data point:
# -> put the values from the data point into the equation y' = 1 / (1 + e^(-wx+b))
# -> choose the label based on the probability

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred

