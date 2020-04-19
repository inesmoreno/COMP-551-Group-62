import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
#x: N
#y: N
plt.plot(x, y, 'b.')
phi = lambda x,mu,sigma: 1/(1 + np.exp(-(x - mu)/(2*sigma*sigma)))
mu = np.linspace(0,3,10)
Phi = phi(x[:,None], mu[None,:]) #N x 10
w = np.linalg.lstsq(Phi, y)[0]
yh = np.dot(Phi,w)
plt.plot(x, yh, 'g-')
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


def relu(x):
    return max(x, 0)


def d_relu(x):
    if x <= 0:
        return 0
    else:
        return 1


def leaky_relu(x, gamma):
    return max(x, 0) + gamma * min(0, x)


def tanh(x):
    return 2 * sigmoid(x) - 1


def softmax(P):
    n, = P.shape
    print(n)
    res = np.zeros((n,))
    tot = 0
    for i in range(n):
        p = np.exp(P[i])
        res[i] = p
        tot += p
    for i in range(n):
        res[i] = res[i] / tot
    return res


class mlp():
    def __init__(self, layers=[1024, 32, 32], input_size=3072, classes=10, activation_function=sigmoid, alpha=0.01,
                 eps=0.01):
        self.Z = []
        self.W = [np.ones((input_size + 1, layers[0])) / input_size]

        for i in range(len(layers) - 1):
            self.Z.append(np.zeros((layers[i] + 1)))
            self.Z[i][layers[i]] = 1
            self.W.append(np.ones((layers[i] + 1, layers[i + 1])) / layers[i])

        self.Z.append(np.zeros((layers[-1] + 1)))
        self.Z[-1][layers[-1]] = 1
        self.W.append(np.ones((layers[-1] + 1, classes)) / layers[-1])

        self.Z.append(np.zeros((classes,)))

        self.D = []
        for i in range(len(layers) - 1):
            D = np.zeros((layers[i], layers[i + 1]))
            self.D.append(D)

        self.activate = activation_function
        self.lr = alpha
        self.eps = eps
        self.num_layers = len(layers)

    def predict(self, input):
        x = torch.cat([input, torch.tensor([1.0])])

        for j in range(len(self.Z[0]) - 1):
            self.Z[0][j] = self.activate(np.dot(x, self.W[0][:, j]))

        for i in range(len(self.Z) - 1):
            for j in range(len(self.Z[i + 1]) - 1):
                self.Z[i + 1][j] = self.activate(np.dot(self.Z[i], self.W[i + 1][:, j]))

        return softmax(self.Z[-1])

    def backprop(self, error):
        ## need the derivation of activation functions
        for i in reversed(range(len(self.D))):
            delta = error * d_sigmoid(
                self.Z[i + 1])  # sigmoid derivative for precedent (change with respect to activation function)
            t_delta = np.reshape(delta.shape[0], -1).T

            active = self.Z[i]
            active = np.reshape(active.shape[0], -1)
            self.D[i] = np.dot(active, delta)
            error = np.dot(delta, self.W[i].T)

    def cross_entropy(self, out, label):
        N = label.shape[0]  # sanity check for num of classes, should be 10
        log_l = -np.log(out[range(N), label])
        ce = np.sum(log_l) / N
        return ce

    def d_cross_entropy(self, out, label):
        N = label.shape[0]
        exp = np.exp(out)
        grad = exp / np.sum(exp)
        grad[range(N), label] -= 1
        delta = grad / N
        return delta

    def stochastic_gradient_descent(self, X):
        """Taken from slides 6.4 gradient descent
        # update the weights by stepping down the gradient
        N, D = X.shape
        w = np.zeros(D)
        g = np.inf
        while np.linalg.norm(g) > self.eps:
            n = np.random.randint(N)
            g = gradient(X[[n],:],y[[n]],w)
            w = w - self.lr*g
        return w """

        for i in range(len(self.W)):
            self.W = self.W[i]
            derivatives = self.D[i]
            self.W += derivatives * self.lr


#doef train():

# error = y - yhat


if __name__ == "__main__":
    perc = mlp()
    # our criterion is cross-entropy
    # our optimizer is sgd

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    main()