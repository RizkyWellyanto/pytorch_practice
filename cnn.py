import torch
import torch.nn as nn
import torchvision.transforms as tf
import torchvision.datasets as dset
from torch.autograd import Variable

# STEP 1: Get Dataset =======================================================
train_dataset = dset.MNIST(root='./data',
                           train=True,
                           transform=tf.ToTensor(),
                           download=True)

test_dataset = dset.MNIST(root='./data',
                          train=False,
                          transform=tf.ToTensor())

# print the dataset size just to check
print(train_dataset.train_data.size())
print(train_dataset.train_labels.size())
print(test_dataset.test_data.size())
print(test_dataset.test_labels.size())

# STEP 2: Make Dataset Iterable ==============================================
batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# STEP 3: Create Model Class =================================================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Max Pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        # Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max Pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max Pool 2
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # New Out Size: (100, 32*7*7)
        out = out.view(out.size(0), -1)

        # Linear function
        out = self.fc1(out)

        return out


# STEP 4: Instantiate Model Class
model = CNNModel()

# USE GPU IF CUDA IS AVAILABLE
if torch.cuda.is_available():
    model.cuda()

# STEP 5: Instantiate Loss Function
loss_func = nn.CrossEntropyLoss()

# STEP 6: Create Optimizer Class
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# STEP 7: Train Model
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as Variable
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        # Clear gradient w.r.t. parameters
        optimizer.zero_grad()

        # Forward Pass to get output
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = loss_func(outputs, labels)

        # Get gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate accuracy
            correct = 0
            total = 0

            # Iterate through test dataset
            for images, labels, in test_loader:
                # Load images to a Torch Variable
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                else:
                    images = Variable(images)

                # Get the output from the model
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
