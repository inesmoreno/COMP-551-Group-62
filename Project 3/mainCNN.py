import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


'''
Compose the ToTensor() transform together with the Normalize transform. Now,
when transform is called in the future it would we first transform the 
image into a PyTorch tensor using ToTensor() and then the tensor image will be normalized
using mean and standard deviation 0.5 for all 3 (RGB colors) channels outputting
tensors of normalized range [-1,1]

'''
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

'''
Creating the train and test sets objects.
The root argument specifies the folder where the sets exist, the train 
argument specifies if either the train set or test set is to be picked, 
in the transform argument  we specify which transform to apply to the data (in this
case, we pass the transform object created above), and, finally the download argument
tells to download the data from an online source  
'''
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

'''
Load the datasets into the dataloader. Here we shuffle the data, separate in in 4 batchs,
'''
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)


testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# functions to show an image (to see what is happening)
def imshow(img):
    # First, unnormalize the image
    img = img / 2 + 0.5
    # Load it into a numpy array
    npimg = img.numpy()
    # Display the immage
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
# and show 4 of them in a grid  and print their labels
#imshow(torchvision.utils.make_grid(images))
#print(''.join('%5s'% classes[labels[j]] for j in range(4)))

# Define the CNN
class Net(nn.Module):
    # Start by defining the layers
    def __init__(self):
        super(Net, self).__init__()
        # First layer: 3 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Pooling layer: 2x2 filter to reduce the dimensionality of the tensor
        self.pool = nn.MaxPool2d(2, 2)

        # Second layer
        self.conv2 = nn.Conv2d(6, 16, 5)

        #Finally, we create fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # Define the forward function through the layers (the backward function is already defined using autograd)
    def forward(self, x):
        # Pass the data (tensor) through the first 2 layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Reshape the tensor so to have 16*5*5 columns and as many rows as needed
        x = x.view(-1, 16 * 5 * 5)

        # Pass the data thought the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# An intance of our model is created
net = Net()

'''
Loss function: we use Cross-Entropy loss, this function combines a SoftMax 
activation and a cross entropy loss function in the same function
'''
criterion = nn.CrossEntropyLoss()

'''
Optimizer: Stochastic Gradient descent, this function takes as argument the 
learnable parameters of the model (net.parameters()) to optimize as well as
the learning rate and the momentum
'''
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# try another optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training our network (2 loops here, maybe can add more)
num_epochs = 2
for epoch in range(num_epochs):  # loop over the data set multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # run the forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # run the backward pass
        loss.backward()
        # optimize
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Saving the trained model for further investigation
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# Check that the model has learnt with images from the test set
dataiter = iter(testloader)
images, labels = dataiter.next()

# show images(with print statement)
imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# Load back saved model
net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

# Run a prediction
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


# Accuracy testing for a set number of images
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

# Accuracy testing for each class
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
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# Train on GPU
device = torch.device ("cuda:0")