import numpy as np


class Regression:
    def __init__(self, method='naive', lmb=0.01):
        self.method = method
        self.lmb = lmb
        self.w = None
        self.x = None
        self.y = None

    def naive_regression(self, x, y):
        m = x.shape[0]
        n = x.shape[1]
        w = np.zeros(n)
        iter_cnt = 0
        learning_rate = 0.1
        err_prev = 9999999
        threshold = 1e-10

        while True:
            y_predict = x.dot(w)
            err = np.sum((y - y_predict) ** 2) / m
            if err_prev - err < threshold:
                print('\n', end='')
                return w
            err_prev = err
            if iter_cnt % 1000 == 0:
                print("\r    Epoch {a}, loss={b}".format(a=iter_cnt, b=err_prev), end="")
            gradient = -2 / m * np.sum(x.transpose() * (y - y_predict), axis=1)
            w = w - learning_rate * gradient
            iter_cnt += 1

    def ridge_regression(self, x, y):
        m, n = x.shape
        w = np.zeros(n)
        iter_cnt = 0
        learning_rate = 0.1
        err_prev = 9999999
        threshold = 1e-10

        while True:
            y_predict = x.dot(w)
            err = np.sum((y - y_predict) ** 2) / m + self.lmb * np.linalg.norm(w, 2) ** 2
            if err_prev - err < threshold:
                print('\n', end='')
                return w
            err_prev = err
            if iter_cnt % 1000 == 0:
                print("\r    Epoch {a}, loss={b}".format(a=iter_cnt, b=err_prev), end="")
            gradient = -2 / m * np.sum(x.transpose() * (y - y_predict), axis=1) + 2 * self.lmb * w
            w = w - learning_rate * gradient
            iter_cnt += 1

    def lasso_regression(self, x, y):
        m = x.shape[0]
        n = x.shape[1]
        w = np.zeros(n)
        iter_cnt = 0
        err_prev = 9999999
        threshold = 1e-9

        while True:
            y_predict = x.dot(w)
            err = np.sum((y - y_predict) ** 2) / m + self.lmb * np.linalg.norm(w, 1)
            if err_prev - err < threshold:
                print('\n', end='')
                return w
            err_prev = err
            if iter_cnt % 100 == 0:
                print("\r    Epoch {a}, loss={b}".format(a=iter_cnt, b=err_prev), end="")
            for k in range(n):
                z_k = (np.transpose(x[:, k]) * x[:, k]).sum()
                w_k = np.transpose(x[:, k]).dot(y - x.dot(w) + w[k] * x[:, k])
                if w_k < -self.lmb / 2:
                    w_k = (w_k + self.lmb / 2) / z_k
                elif w_k > self.lmb / 2:
                    w_k = (w_k - self.lmb / 2) / z_k
                else:
                    w_k = 0
                w[k] = w_k
            iter_cnt += 1

    def train(self, x, y):
        if self.method == 'naive':
            self.w = self.naive_regression(x, y)
        elif self.method == 'ridge':
            self.w = self.ridge_regression(x, y)
        elif self.method == 'lasso':
            self.w = self.lasso_regression(x, y)

    def predict(self, x):
        return x.dot(self.w)
