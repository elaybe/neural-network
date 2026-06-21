import numpy as np


class output_layer():
    def __init__(self):
        self.weights = None
        self.bias = None

    def forwarding(self, x):        
        if self.weights is None:                        
            w_dim = x.shape[1]
            n_dim = x.shape[0]
            sigma = np.sqrt(2 / (w_dim + n_dim))
            self.weights = np.random.normal(loc=0, scale=sigma, size=[w_dim, 1])
            self.bias = np.zeros((1, 1))

        z= np.dot(x, self.weights) + self.bias
        a= 1 / (1 + np.e ** -z)
        return a
    

class NeuralNetwork:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.ol = None

    def forwarding(self, x):        
        if self.weights is None:
            w_dim = x.shape[1]
            n_dim = 2 ** w_dim
            sigma = np.sqrt(2 / (x.shape[1] + x.shape[0]))
            self.weights = np.random.normal(loc=0, scale=sigma, size=[w_dim,n_dim])
            self.bias = np.zeros((1, n_dim))
            self.ol = output_layer()

        z= np.dot(x, self.weights) + self.bias
        a= 1 / (1 + np.e ** -z)
        return self.ol.forwarding(a)
    
if __name__ == "__main__":
    X_all = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])

    nn = NeuralNetwork()
    prediction = nn.forwarding(X_all)

    print(prediction)