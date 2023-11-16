from Layer import Layer

class Classifier:
    def __init__(self):
        self.layer1 = Layer(64)
        self.layer2 = Layer(64)
        self.layer3 = Layer(10)
    
    
    def forward(self, inputData):
        pass