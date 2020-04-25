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
when the new transform is used in the future it would first transform the 
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
Load the datasets into the dataloader. Here we only shuffle the data in the trainset
and batch the data from both sets.
'''
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
 Define the CNN
 The network hs the following layout:
 Input > Conv (ReLU) > MaxPool > Conv (ReLU) >
 MaxPool > FC (ReLU) > FC (ReLU) > FC (SoftMax) > 10 outputs
 where: Conv is a convolutional layer, ReLu is the activation function, MaxPool
 is a pooling layer, FC is a fully connected layer and SoftMax is the activation 
 of the output layer
'''


class Net(nn.Module):
    # Start by defining the layers
    def __init__(self, num_filters1, num_filters2, kernel_size, activation):
        super(Net, self).__init__()
        '''
        First layer: 
        The input images have 3 channels (RGB) of size 32x32px each
        Hence, this layer expects 3 input channels (first argument). It 
        will use 6 convolutional filters (second argument) each of size 5x5 (third argument)
        The padding is set to 0 (and hence the stride is 1).
        The output size will then be 6x28x28 (because (32-5)+1 =28). 
        '''
        self.conv1 = nn.Conv2d(3, num_filters1, kernel_size)

        '''
        Pooling layer: reduces the dimensionality of the tensor.
        Here we use max pooling with a 2x2 filter (first argument) and
        stride 2 (second argument).
        '''
        self.pool = nn.MaxPool2d(2, 2)

        '''
        Second layer:
        This layer expects 6 input channels (first argument), it uses 16 convolutional
        filters (second argument) each of size 5x5 (third argument). The padding is set
        to 0 again so stride will be 1 which gives us an output of size 16x10x10.
        '''
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size)

        # Can specify a dropout (regularization) layer  to avoid over-fitting the model
        # self.drop_out = nn.Dropout()

        '''
        Finally, we create  fully connected  linear layers.
        First we connect a layer of 16*5*5 nodes (first argument) to a
        layer of 120 nodes (second argument).
        '''
        self.fc1 = nn.Linear(num_filters2 * kernel_size * kernel_size, 120)

        '''
        Then we connect a layer of 120 nodes (first argument) to a layer of
        84 nodes (second argument)
        '''
        self.fc2 = nn.Linear(120, 84)

        '''
        Finally we connect a layer of 84 nodes to the final (output) layer of 10 nodes.
        '''
        self.fc3 = nn.Linear(84, 10)

        self.activate = activation
        self.kernel = kernel_size
        self.filter1 = num_filters1
        self.filter2 = num_filters2

    # Define the forward function through the layers (the backward function is already defined using autograd)
    def forward(self, x):
        '''
        Pass the data (tensor) through the first convolutional layer: here we use
        ReLu as our activation function. And the first pooling layer (using the max pool
        defined above). The output size of the convolutional layer is 6x28x28 and after passing
        it through the pooling we get a size of 6x14x14.
        '''
        x = self.pool(self.activate(self.conv1(x)))

        '''
        Pass the data (tensor) through the second convolutional layer: here we use
        ReLu as our activation function again. And a second pooling layer (using the max pool
        defined above). The output size of the convolutional layer is 16x10x10 and after passing
        it through the pooling layer we get a size of 16x5x5.
        '''
        x = self.pool(self.activate(self.conv2(x)))

        '''
        Need to reshape the output from the last pooling layer to connect it to the 
        fully connected layer: we need to have 16*5*5 columns (second argument) and as
        many rows as needed (first argument -1: automatically infers the number of rows
        required)
        '''
        x = x.view(-1, self.filter2 * self.kernel * self.kernel)

        # Then, the drop_out is applied
        # x = self.drop_out(x)

        # Pass the data though the 2 fully connected layers using ReLU as activation function
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))

        # Pass the data through the output layer
        x = self.fc3(x)
        return x


def train(cnn, optimizer, lr, epochs):
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
    opt = optimizer(cnn.parameters(), lr=lr)

    '''
    Training our network: go over the training data in batches of <batch_size> images and
    repeat the whole process <epochs> times
    '''
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # run the forward pass
            outputs = cnn(inputs)
            # compute the loss
            loss = criterion(outputs, labels)
            # run the backward pass (backpropagation)
            loss.backward()
            # optimize
            opt.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


# functions to show an image (to see what is happening)
def imshow(img):
    # First, unnormalize the image
    img = img / 2 + 0.5
    # Load it into a numpy array
    npimg = img.numpy()
    # Display the immage
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


######################################################################
### Analysing the effect of the hyperparameters on the performance ###
######################################################################

def compare_optimizer(cnn, opti=[optim.Adam, optim.SGD]):
    lr_list = np.linspace(0.001, 0.003, 5)  # List of lr values to test
    acc_list = []  # List storing results
    best_acc = 0
    for o in opti:
        if o == optim.Adam:
            for l in lr_list:
                train(cnn, o, l, 5)
                acc = test_classes()
                if acc > best_acc:
                    best_acc = acc
                acc_list.append(acc)

            # Plot the results
            plt.plot(lr_list, acc_list, 'ro')
            plt.xlabel('Learning rate')
            plt.ylabel('Accuracy')
            plt.title('ADAM (Accuracy vs. Learning rate)')
            plt.show()
            acc_list = []  # Restart the list to plot the second optimizer
        else:
            for l in lr_list:
                train(net, o, l, 5)
                acc = test_classes()
                if acc > best_acc:
                    best_acc = acc
                acc_list.append(acc)

            # Plot the results
            plt.plot(lr_list, acc_list, 'ro')
            plt.xlabel('Learning rate')
            plt.ylabel('Accuracy')
            plt.title('SGD (Accuracy vs. Learning rate)')
            plt.show()


def compare_epochs(net):
    acc_list = []  # List storing results
    best_acc = 0
    for e in range(2, 11):
        train(net, optimizer=optim.Adam, lr=0.0015, epochs=e)
        acc = test_classes()
        if acc > best_acc:
            best_acc = acc
        acc_list.append(acc)

    # Plot the results
    plt.plot(range(2, 11), acc_list, 'ro')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('(Accuracy vs. Epochs)')
    plt.show()

def compare_layer_filters():
    num_filter = [8, 16, 32, 64]
    acc_list = []  # List storing results
    best_acc = 0
    for i in range(3):
        net = Net(num_filters1=num_filter[i], num_filters2=num_filter[i+1], kernel_size=5, activation=F.relu)
        train(net, optimizer=optim.Adam, lr=0.0015, epochs=5)
        acc = test_classes()
        if acc > best_acc:
            best_acc = acc
        acc_list.append(acc)


#net= Net(num_filters1=6, num_filters2=16, kernel_size=5, activation=F.relu)
#net = Net(num_filters1=6, num_filters2=16, kernel_size=5, activation=F.leaky_relu)

'''
# Test the model
dataiter = iter(testloader)
images, labels = dataiter.next()


Input test images to the model to get the class labels. The model outputs a 2D tensor
of size <batch_size>x10, a row for each image in the batch and a column for each category. 
The category predicted for each image will then be the column index with the maximum value 
in that row. 

outputs = net(images)

# Run a prediction
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


# Accuracy testing for a set number of images
correct = 0
total = 0
#net.eval() #disables drop-out and normalization layers
with torch.no_grad(): #disables autograd
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
'''


def test_classes():
    # Accuracy testing for each class
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    confusion_matrix = np.zeros([10, 10], int)
    # net.eval()
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
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1
    av_acc = 0
    for i in range(10):
        av_acc += class_correct[i] / class_total[i]
    av_acc = av_acc / 10
    print('Average accuracy %.2f' % av_acc)
    '''
    # Visualize confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
    plt.ylabel('Actual Category')
    plt.yticks(range(10), classes)
    plt.xlabel('Predicted Category')
    plt.xticks(range(10), classes)
    plt.show()
    '''
    return av_acc


# compare_optimizer(net)
#compare_epochs(net1)
#train(net, optimizer=optim.Adam, lr=0.0015, epochs=5)
#test_classes()
compare_layer_filters()



'''
# Visualize confusion matrix
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
plt.ylabel('Actual Category')
plt.yticks(range(10), classes)
plt.xlabel('Predicted Category')
plt.xticks(range(10), classes)
plt.show()
'''
