from random import randint
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transform
import matplotlib.pyplot as plt

class DataLoader:

    data_path = "./data"

    #Loading the data and splitting them into test and train data
    global mnist_trainset, mnist_testset, x_train, y_train, x_test, y_test, labels, histogram

    mnist_trainset = datasets.MNIST(
        root=data_path, 
        train=True, 
        download=True, 
        transform=transform.ToTensor()
    )

    mnist_testset = datasets.MNIST(
        root=data_path, 
        train=False, 
        download=True, 
        transform=transform.ToTensor()
    )

    (x_train, y_train) = mnist_trainset[0]
    (x_test, y_test) = mnist_testset[0]

    labels = []
    histogram = []

    #Initializes the arrays
    for i in range(10):
        labels.append(i)

    for i in range(10):
        histogram.append(i)

    #Caluclates the spread of the data
    for i in range(len(mnist_trainset)):
        img, label = mnist_trainset[i]
        histogram[label] += 1

    
    #scales the input according to the given formula
    def normalization(self, data: np.ndarray):
        min, max = np.min(data), np.max(data)
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = 2 * (data[i][j] - min) / (max - min) -1
        return data

    
    def getSampleData(self):
        random = randint(0, len(mnist_trainset))
        data = mnist_trainset.data[random].float().numpy()
        print(data.ndim)
        data = self.normalization(data)
        return data

    def getNormalizedTrainData(self):
        normalize_data = []
        for i in range (len(mnist_trainset.data)):
            data = mnist_trainset.data[i].float().numpy()
            data = self.normalization(data)
            normalize_data.append(data)
        
        return normalize_data
        

    
    
    def getNormalizedTestData(self):
        normalize_data = []
        for i in range (len(mnist_trainset.data)):
            data = mnist_testset.data[i].float().numpy()
            data = self.normalization(data)
            normalize_data.append(data)
        
        return normalize_data



    