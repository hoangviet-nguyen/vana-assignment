from Layer import Layer
import numpy as np

class Classifier:
    def __init__(self):
       self.layer1 = Layer(16, 784)
       self.layer2 = Layer(16, 16)
       self.layer3 = Layer(16, 16)
       self.layer4 = Layer(10, 16)
    
    
    def forward(self, inputData):
        #sending the input through the layers

        self.layer1.linear(inputData)
        x = self.layer1.applySigmoid()

        self.layer2.linear(x)
        x = self.layer2.applySigmoid()

        self.layer3.linear(x)
        x = self.layer3.applySigmoid()

        #at the end we want to classify our data so we are using the softmax function
        self.layer4.linear(x)
        x = self.layer4.applySoftmax()
        
        return x
    
    #here we are calculating the loss with the cross entropy loss function
    def compute_loss(y_target: np.ndarray, y_predicted: np.ndarray):
        sum = 0

        for i in range(len(y_predicted)):
            sum += y_target[i] * np.log(y_predicted[i]) + (1-y_target[i]) * np.log(1 - y_predicted[i])
    
        sum /= -len(y_predicted)

        return sum
    
    