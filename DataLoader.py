from random import randint
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transform

class DataLoader:

    data_path = "./data"

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
    
    #scales the input according to the given formula
    def normalization(self, data: np.ndarray):
        min, max = np.min(data), np.max(data)
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = 2 * (data[i][j] - min) / (max - min) -1
        return data

    
    def getSampleData(self):
        random = randint(0, len(self.mnist_trainset))
        data = self.mnist_trainset.data[0].float().numpy()
        data = self.normalization(data)
        return data.flatten()

    def getNormalizedTrainData(self):
        normalize_data = []
        for i in range (len(self.mnist_trainset.data)):
            data = self.mnist_trainset.data[i].float().numpy()
            data = self.normalization(data)
            normalize_data.append(data)
        
        return normalize_data
        
    
    def getNormalizedTestData(self):
        normalize_data = []
        for i in range (len(self.mnist_trainset.data)):
            data = self.mnist_testset.data[i].float().numpy()
            data = self.normalization(data)
            normalize_data.append(data)
        
        return normalize_data



    