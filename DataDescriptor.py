import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transform
import matplotlib.pyplot as plt

class DataDescriptor:

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

    
    
    #Desribes the data we are working with
    def data_format():
        print("The size of the training dataset is: " + str(len(mnist_trainset)))
        print("The size of the test set is: " + str(len(mnist_testset)))
        print("The size of one image is: " + str(mnist_trainset.data.shape))
        print("The Input container is: " + str(mnist_trainset.data[0]))
        print("The data types of the input are: " + str(type(y_train)) + " and " + str(type(x_train)))
        print("Der Input Wertebereich ist: ", f"[{mnist_trainset.data.min()}, {mnist_trainset.data.max()}]")


    
    #Visualizes some sample data with their labels
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

    #Visualize the spread of the data
    def createHistorgramm():
        plt.bar(labels, histogram, edgecolor='black')
        plt.xlabel("Label")
        plt.ylabel("Number of images")
        plt.title("Spread of the data")
        plt.show()