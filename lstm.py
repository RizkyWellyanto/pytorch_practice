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
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building RNN
        # batch_first=True causes input/output tensor to be of shape (batch_dim, seq_dim, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want las time step in hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10

        return out

# STEP 4: Instantiate Model Class
input_dim = 28
hidden_dim = 100
layer_dim = 1
output_dim = 10

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

# USE GPU IF CUDA IS AVAILABLE
if torch.cuda.is_available():
    model.cuda()

# STEP 5: Instantiate Loss Function
loss_func = nn.CrossEntropyLoss()

# STEP 6: Create Optimizer Class
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# STEP 7: Train Model

# Number of steps to unroll
seq_dim = 28

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as Variable
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_dim, input_dim).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, seq_dim, input_dim))
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
                    images = Variable(images.view(-1, seq_dim, input_dim).cuda())
                else:
                    images = Variable(images.view(-1, seq_dim, input_dim))

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





