import numpy as np
from Node import Node

class Layer:
    #Initialize the layer with abritrary nodes
    def __init__(self, numberOfNodes: int, inputData: np.ndarray,):
        self.numberOfNodes = numberOfNodes
        self.nodes = [Node() for _ in range(numberOfNodes)]
        self.output = []
        self.inputData = inputData
        self.weights = np.random.uniform(low = -1, high = 1, size=(numberOfNodes, len(inputData)))
        self.bias = np.random.uniform(low = -5, high= 5, size=(numberOfNodes))

    #apply the linear transformation to each node
    def linear(self):
        for i in range(self.numberOfNodes):
            node = self.nodes[i]
            transformed = node.linear_transformation(self.inputData, self.weights[i], self.bias[i])
            self.output.append(transformed)    

    def applySigmoid(self):
        for i in range(self.numberOfNodes):
            self.output[i] = self.sigmoid(self.output[i])

        return self.output
        
    
    def applyRelu(self):
        for i in range (self.numberOfNodes):
            self.output[i] = self.Relu(self.output[i])

        return self.output
    
    def applySoftmax(self):
        sumExpo = sum(np.exp(self.output[i]) for i in range(len(self.output)))
        for i in range(len(self.output)):
            self.output[i] = self.softmax(self.output[i], sumExpo)

        return self.output

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))
    
    def Relu(self, value):
        return value if value > 0 else 0
    

    def softmax(self, value, sumExpo):
        return np.exp(value) / sumExpo