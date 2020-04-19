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

def relu(x):
    return max(x, 0)

def leaky_relu(x, gamma):
    return max(x, 0) + gamma*min(0, x)

def tanh(x):
    return 2*sigmoid(x) - 1


def softmax(P):
    n, = P.shape
    res = np.zeros((n,))
    tot = 0
    for i in range(n):
        p = np.exp(P[i])
        res[i] = p
        tot += p
    for i in range(n):
        res[i] = res[i]/tot
    return res

class mlp():
    def __init__(self, layers=[1024, 32, 32], input_size=3072, classes=10, activation_function=sigmoid,
                alpha=0.01, eps=0.01, batch_size=10):
        self.Z = []
        self.W = [np.ones((input_size+1, layers[0]))/input_size]
        
        for i in range(len(layers) - 1):
            self.Z.append(np.zeros((layers[i]+1)))
            self.Z[i][layers[i]] = 1
            self.W.append(np.ones((layers[i]+1, layers[i+1]))/layers[i])

        self.Z.append(np.zeros((layers[-1]+1)))
        self.Z[-1][layers[-1]] = 1
        self.W.append(np.ones((layers[-1]+1, classes))/layers[-1])

        self.Z.append(np.zeros((classes,)))

        self.activate = activation_function
        self.lr = alpha
        self.eps = eps


    def predict_proba(self, input):
        x = torch.cat([input, torch.tensor([1.0])])

        for j in range(len(self.Z[0])-1):
            self.Z[0][j] = self.activate(np.dot(x, self.W[0][:, j]))

        for i in range(len(self.Z)-1):
            for j in range(len(self.Z[i+1])-1):
                self.Z[i+1][j] = self.activate(np.dot(self.Z[i], self.W[i+1][:, j]))

        return softmax(self.Z[-1])

    def predict(self, input):
        yh = self.predict_proba(input)
        print(yh)
        return np.argmax(yh)


perc = mlp()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


test = []

def add_sym(dataset):
    res = []
    for d in dataset:
        res.append(d)
        res.append((torch.flip(d[0], [2]), d[1]))
    return res


def add_noise(dataset, noise=0.1):
    res = []
    for d in dataset:
        res.append(d)
        s = d[0].size
        res.append((d[0] + 2*noise*(torch.rand(s) - 0.5), d[1]))
    return res

#foo = add_sym(trainset)


for i in range(5):
    test.append(trainset[i][0])
    s = trainset[i][0].size()
    test.append(trainset[i][0] + 2*0.1*(torch.rand(s) - 0.5))
    #test.append(torch.flip(trainset[i][0], [2]))

    #x = trainset[i][0].reshape(3072)
    #print(x)
    #print(perc.predict(trainset[i][0].reshape(3072)))
    #print(np.dot(trainset[i][0].reshape(3072), np.ones((3072, ))))
    #print(activate(trainset[i][0].reshape(3072), np.ones((3072, ))/3072))


for im in test:
    print(im)
    imshow(im)


"""
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
"""
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

net.to(device)

inputs, labels = data[0].to(device), data[1].to(device)
"""

