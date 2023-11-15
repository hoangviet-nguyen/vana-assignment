import numpy as np
from Node import Node

class Layer:
    nodes = []
    output = []

    #Initialize the layer with abritrary nodes
    def __init__(self, numberOfNodes: int):
        self.numberOfNodes = numberOfNodes
        
        #Initialize the nodes and storing them for later computation
        for i in range(numberOfNodes):
            node = Node()
            self.nodes.append(node)


    #apply the linear transformation to each node
    def forward(self, inputData: np.ndarray, weights: np.ndarray, bias: float):
        for i in range(self.numberOfNodes):
            node = self.nodes[i]
            transformed = node.linear_transformation(inputData, weights[i], bias)
            self.output.append(transformed)    

    #apply the sigmoid function for the end output
    def applySigmoid(self):
        for i in range(self.numberOfNodes):
            self.output[i] = self.sigmoid(self.output[i])
        
        print(self.output)


    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))