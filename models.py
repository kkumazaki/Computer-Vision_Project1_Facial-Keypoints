## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # I chose NaimishNet, which was developed for Facial Keypoints Detection problem.
        # https://arxiv.org/pdf/1710.00977.pdf
        
        # Parameters
        # Try1: The result was overfitting...
        #p1 = 0.1
        #p2 = 0.2
        #p3 = 0.3
        #p4 = 0.4
        #p5 = 0.5
        #p6 = 0.6

        p1 = 0.5
        p2 = 0.5
        p3 = 0.5
        p4 = 0.5
        p5 = 0.5
        p6 = 0.5

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), output channels/feature maps, square convolution kernel
        # Input Image size: [1, 224, 224], Layer Shape: [32, 221, 221]
        self.conv1 = nn.Conv2d(1, 32, 4)

        # 2 Maxpooling
        # Layer Shape: [32, 110, 110]
        self.pool1 = nn.MaxPool2d(2, 2)  

        # 3 dropout with p=0.5
        # Layer Shape: [32, 110, 110]        
        self.drop1 = nn.Dropout(p=p1)

        # 4 input image channel (grayscale), output channels/feature maps, square convolution kernel
        # Layer Shape: [64, 108, 108]
        self.conv2 = nn.Conv2d(32, 64, 3)

        # 5 Maxpooling
        # Layer Shape: [64, 54, 54]
        self.pool2 = nn.MaxPool2d(2, 2)  

        # 6 dropout with p=0.5
        # Layer Shape: [64, 54, 54]
        self.drop2 = nn.Dropout(p=p2)

        # 7 input image channel (grayscale), output channels/feature maps, square convolution kernel
        # Layer Shape: [128, 53, 53]
        self.conv3 = nn.Conv2d(64, 128, 2)

        # 8 Maxpooling
        # Layer Shape: [128, 26, 26]
        self.pool3 = nn.MaxPool2d(2, 2)

        # 9 dropout with p=0.5
        # Layer Shape: [128, 26, 26]
        self.drop3 = nn.Dropout(p=p3)

        # 10 input image channel (grayscale), output channels/feature maps, square convolution kernel
        # Layer Shape: [256, 26, 26]
        self.conv4 = nn.Conv2d(128, 256, 1)

        # 11 Maxpooling
        # Layer Shape: [256, 13, 13]
        self.pool4 = nn.MaxPool2d(2, 2)

        # 12 dropout with p=0.5
        # Layer Shape: [256, 13, 13]
        self.drop4 = nn.Dropout(p=p4)

        # 13 outputs * the 5*5 filtered/pooled map size
        # Layer Shape: [256 * 13 * 13] --> [1000]
        self.fc1 = nn.Linear(256*13*13, 1000)
        # Layer Shape: [64 * 54 * 54] --> [10000]
        #self.fc1 = nn.Linear(64*54*54, 10000) # too big and memory error

        # 14 dropout with p=0.5
        # Layer Shape: [1000]
        self.drop5 = nn.Dropout(p=p5)

        # 15 outputs * the 5*5 filtered/pooled map size
        # Layer Shape: [1000] --> [1000]
        #self.fc2 = nn.Linear(1000, 1000)
        # Layer Shape: [10000] --> [1000]
        self.fc2 = nn.Linear(10000, 1000) # no need after changing fc1

        # 1:6 dropout with p=0.5
        # Layer Shape: [1000]
        self.drop6 = nn.Dropout(p=p6)

        # 17 outputs * the 5*5 filtered/pooled map size
        # Layer Shape:  [1000] --> [2*68]
        self.fc3 = nn.Linear(1000, 136)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # conv/relu + pool layers
        x = self.pool1(F.relu(self.conv1(x)))

        # dropout
        x = self.drop1(x)

        # conv/relu + pool layers
        x = self.pool2(F.relu(self.conv2(x)))       

        # dropout
        x = self.drop2(x)

        # conv/relu + pool layers
        x = self.pool3(F.relu(self.conv3(x)))

        # dropout
        x = self.drop3(x)

        # conv/relu + pool layers
        x = self.pool4(F.relu(self.conv4(x)))       

        # dropout
        x = self.drop4(x)

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.drop5(x)

        # linear layers with dropout in between
        #x = F.relu(self.fc2(x))
        #x = self.drop6(x)

        # finally, create 2 output channels (for the 2 classes of Keypoint: x, y)
        x = self.fc3(x) 
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
