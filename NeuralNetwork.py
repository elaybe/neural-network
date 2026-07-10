import numpy as np

class layer():
    def __init__(self, x, y, neurons):
        self.x = x
        self.y = y
        self.neurons = neurons
        self.weights = None
        self.bias = None
        self.a = None
        
    def __initializing(self, x):
        if self.weights is None:
            col = x.shape[1]
            sigma = np.sqrt(2 / (col + self.neurons))
            self.weights = np.random.normal(loc=0, scale=sigma, size=[col, self.neurons])
            self.bias = np.zeros((1, self.neurons))

    def forwarding(self, layers):
        x = self.x
        for neuron in layers:
            neuron.__initializing(x)
            z = np.dot(x, neuron.weights) + neuron.bias
            neuron.x = x
            neuron.a = x = 1 / (1 + np.e ** -z)

        return neuron.backpropagation(layers)

    def backpropagation(self, layers):
        learning_rate = 0.01

        for layer in reversed(layers):

            if layer is self:
                dl_da = layer.a - layer.y
                da_dz = layer.a * (1 - layer.a)
                dz_da = layer.weights
                dz_dw = layer.x

                dl_dz = dl_da * da_dz
                delta = dl_dz @ dz_da.T
                gradient_equation = dl_dz.T @ dz_dw
                db = np.sum(dl_dz, axis=0, keepdims=True)
                loss = 1/2 * np.mean((layer.y - layer.a) ** 2)

            else:
                da_dz = layer.a * (1 - layer.a)
                dz_da = layer.weights.T
                dz_dw = layer.x
                gradient_equation = delta.T @ dz_dw
                db = np.sum(delta, axis=0, keepdims=True)
                delta = delta * da_dz @ dz_da 
                
            layer.weights = layer.weights - learning_rate * gradient_equation.T
            layer.bias = layer.bias - learning_rate * db
            
        return loss

if __name__ == "__main__":
    x = np.array([
        [0.1, 0.2, 0.7, 0.9],
        [0.5, 0.4, 0.1, 0.2],
        [0.8, 0.9, 0.3, 0.4]
    ])

    y = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    ])
        
    hidden_layer1 = layer(x, y, neurons=30)
    hidden_layer2 = layer(x, y, neurons=30)
    hidden_layer3 = layer(x, y, neurons=30)
    output_layer = layer(x, y, neurons=10)

    layers = np.array([hidden_layer1, hidden_layer2, hidden_layer3, output_layer])

    for epoch in range(1000):
        loss = hidden_layer1.forwarding(layers)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
