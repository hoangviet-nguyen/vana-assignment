import numpy as np

class Node:
        
    def linear_transformation(self, inputData: np.ndarray, weights: np.ndarray, bias: float):
        return np.dot(inputData, weights) + bias