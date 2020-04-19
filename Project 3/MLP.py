import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

def d_leaky_relu(x, gamma):
    if x <= 0:
        return gamma
    else:
        return 1


def tanh(x):
    return 2 * sigmoid(x) - 1

def d_tanh(x):
    return 1 - tanh(x)**2


def softmax(P):
    n, = P.shape
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
    def __init__(self, layers=[1024, 32, 32], input_size=3072, classes=10, activation_function=sigmoid, deriv=d_sigmoid,
                 alpha=0.05, eps=0.25, batch_size=8):
        self.Z = [np.zeros((input_size + 1,))]              # Z contains the value associated with each neuron
        self.W = [np.zeros((input_size + 1, layers[0]))]    # W contains the weigth between each neuron

        for i in range(len(layers) - 1):
            self.Z.append(np.zeros((layers[i] + 1,)))
            self.Z[i][layers[i]] = 1                        # Add a neuron for the bias
            self.W.append(np.zeros((layers[i] + 1, layers[i + 1])))

        self.Z.append(np.zeros((layers[-1] + 1,)))
        self.Z[-1][layers[-1]] = 1
        self.W.append(np.zeros((layers[-1] + 1, classes)))

        self.Z.append(np.zeros((classes,)))

        # The derivatives will be stored in D
        self.D = [np.zeros((input_size + 1, layers[0]))]
        for i in range(len(layers) - 1):
            D = np.zeros((layers[i] + 1, layers[i + 1]))
            self.D.append(D)
        self.D.append(np.zeros((layers[-1] + 1, classes)))

        self.activate = activation_function
        self.derivative = deriv
        self.lr = alpha
        self.eps = eps
        self.batch_size = batch_size
        self.num_layers = len(layers)

    def predict_proba(self, input):
        # Takes an image as input and return an array of the predicted probabilities of being in a given class

        # Unravel the input into a 1-D tensor, and add a last feature for the bias
        s = input.size()
        N = 1
        for n in s:
            N = N * n
        x = torch.cat([input.reshape(N), torch.tensor([1.0])])

        self.Z[0] = np.array(x)

        for i in range(len(self.Z)-2):
            # Compute the activity of the next layer
            for j in range(len(self.Z[i+1])-1):
                self.Z[i+1][j] = self.activate(np.dot(self.Z[i], self.W[i][:, j]))

        # For the last layer, replace the activation function by a softmax function
        A = np.zeros((len(self.Z[-1])))
        for j in range(len(A)):
            A[j] = np.dot(self.Z[-2], self.W[-1][:, j])

        self.Z[-1] = softmax(A)

        return self.Z[-1]

    def predict(self, input):
        # Takes an image as input and return a predicted label
        yh = self.predict_proba(input)
        return np.argmax(yh)

    def backprop(self, yh, label):
        ## need the derivation of activation functions
        error = yh
        error[label] += -1
        delta = np.copy(error)

        self.D[-1] = np.outer(delta.T, self.Z[-2]).T


        for i in reversed(range(len(self.D)-1)):
            error = np.dot(delta, self.W[i+1].T)
            delta = np.delete(error, len(error)-1)
            for j in range(len(delta)-1):
                delta[j] = delta[j] * self.derivative(np.dot(self.Z[i], self.W[i][:, j]))  # sigmoid derivative for precedent (change with respect to activation function)
            self.D[i] = np.outer(delta.T, self.Z[i]).T


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

    def batch_stochastic_gradient_descent(self, X):
        N = len(X)
        g = np.inf

        # While the gradient is high enough, repeat to reach convergence
        while g > self.eps:
            # Create a nameplace where we will store the gradient
            grad = []
            for l in range(len(self.D)):
                s = self.D[l].shape
                grad.append(np.zeros(s))

            # Randomly select a batch of the training set of the desired size
            batch = random.sample(range(N), self.batch_size)
            # For each sample of the batch, compute the associated gradient with backpropagation
            for i in batch:
                yh = self.predict_proba(X[i][0])
                
                self.backprop(yh, X[i][1])
                for l in range(len(self.D)):
                    grad[l] += self.D[l]
            
            g = 0
            for l in range(len(self.D)):
                # Get the average of the gradient
                grad[l] = grad[l] / self.batch_size
                g += np.linalg.norm(grad[l])
                # Update the weights
                self.W[l] += -grad[l] * self.lr
            print("g : ", g)

    def fit(self, trainset):
        # Use the mini-batch stochastic gradient descent to fit the weigths of the model to the training set
        self.batch_stochastic_gradient_descent(trainset)




def get_sym(img):
    # Return the symetric of the image
    return torch.flip(img, [2])

def add_noise(img, noise=0.1):
    # Return the superposition of the image with a noise of range 'noise'
    s = img.size()
    return img + 2*noise*(torch.rand(s) - 0.5)

def get_transpose(img):
    # Return the rotated version of the image
    return torch.transpose(img, 1, 2)

def add_data(dataset, sym=True, transpose=True, noise=0.1, noise_iter=1):
    # Add new images to the dataset to avoid overfitting
    res = []
    for d in dataset:
        res.append(d)
        if sym:
            d_sym = get_sym(d[0])
            res.append((d_sym, d[1]))
            if transpose:
                d_st = get_transpose(d_sym)
                res.append((d_st, d[1]))
                if noise > 0:
                    for i in range(noise_iter):
                        res.append((add_noise(d_st, noise), d[1]))

        if transpose:
            d_t = get_transpose(d[0])
            res.append((d_t, d[1]))
            if noise > 0:
                for i in range(noise_iter):
                    res.append((add_noise(d_t, noise), d[1]))

        if noise > 0:
            for i in range(noise_iter):
                res.append((add_noise(d[0], noise), d[1]))

    return res


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


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

    aug_train = add_data(trainset, False, False, 0)


    perc.fit(aug_train)

