#Followed this tutorial for loading the data https://www.datascienceweekly.org/tutorials/pytorch-mnist-load-mnist-dataset-from-pytorch-torchvision

import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transform
import matplotlib.pyplot as plt
from PIL import Image

data_path = "./data"

#defining the data and splitting them 
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

#loading the data
(x_Image, y_trainInput) = mnist_trainset._load_data()
x_Image, y_testInput = mnist_testset._load_data()

labels = []
histogram = []

for i in range(10):
    labels.append(i)

for i in range(10):
    histogram.append(i)


#Desribes the data we are working with
def data_format():
    print("The size of the training dataset is: " + len(mnist_trainset))
    print("The size of the test set is: " + len(mnist_trainset))
    print("The size of one image is: " + len(x_Image) * len(x_Image))
    print("The Input container is: " + type(mnist_trainset[0]))
    print("The data types of the input are: " + type(y_trainInput[0]) + " and " + type(x_Image[0]))


def visualizeData():
    figure = plt.figure(figsize=(16, 16))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        index = torch.randint(len(mnist_trainset), size=(1,)).item()
        img, label = mnist_trainset[index]
        figure.add_subplot(rows, cols, i)
        plt.title("labels = " + str(labels[label]))
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

def createHistorgramm():
    for i in range(len(mnist_trainset)):
        img, label = mnist_trainset[i]
        histogram[label] += 1
    
    plt.bar(labels, histogram, edgecolor='black')
    plt.xlabel("Label")
    plt.ylabel("Number of images")
    plt.title("Spread of the data")
    plt.show()

if __name__ == "__main__":
    createHistorgramm()
    visualizeData()