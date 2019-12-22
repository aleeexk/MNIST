from keras.datasets import mnist
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()


class NN:
    def __init__(self, rate, inputs, hiddens, outputs):
        self.i = inputs + 1
        self.h = hiddens
        self.o = outputs
        self.ihw = np.random.normal(0.0, pow(self.h, -0.5), (self.h, self.i))
        self.how = np.random.normal(0.0, pow(self.o, -0.5), (self.o, self.h))
        self.lr = rate
        self.sigmoid = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs_list = np.concatenate((inputs_list, [1]), axis=0)
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hid_results = self.sigmoid(np.dot(self.ihw, inputs))
        out_results = self.sigmoid(np.dot(self.how, hid_results))
        out_errors = (targets - out_results)
        hid_errors = np.dot(self.how.T, out_errors)
        self.how += self.lr * np.dot(out_errors * out_results * (1.0 - out_results), np.transpose(hid_results))
        self.ihw += self.lr * np.dot(hid_errors * hid_results * (1.0 - hid_results), np.transpose(inputs))

    def query(self, inputs_list):
        inputs_list = np.concatenate((inputs_list, [1]), axis=0)
        inputs = np.array(inputs_list, ndmin=2).T
        hid_result = self.sigmoid(np.dot(self.ihw, inputs))
        out_results = self.sigmoid(np.dot(self.how, hid_result))
        return out_results

    def set_lr(self, rate):
        self.lr = rate


n = 5999
plt.imshow((255 - x_test[n]) / 255, cmap="gray")
plt.show()
network = NN(0.1, 784, 100, 10)
m = int(input())
i = 1


def train(n):
    target = np.zeros(10)
    target[y_train[n]] = 1
    query = np.array(x_train[n] / 255).reshape(784)
    network.train(query, target)


def test(n):
    query = np.array(x_test[n] / 255).reshape(784)
    return network.query(query)


def epoch_train():
    network.set_lr(0.1)
    x_train_len = len(x_train)
    for n in range(x_train_len):
        train(n)
        if n % 10000 == 0:
            print("Row: %s\r" % n)


def epoch_test():
    x_test_len = len(x_test)
    presicion = 0
    for n in range(x_test_len):
        ans = test(n)
        if ans.argmax() == y_test[n]:
            presicion += 1
    return presicion / (n + 1)


while i < m:
    epoch_train()
    i = i + 1
print(test(n).argmax())

print(epoch_test())
