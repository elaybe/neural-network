"""Compare backpropagation gradients with numerical gradients.

Run with: python test_gradients.py
"""

import sys

import numpy as np

from NeuralNetwork import layer

EPSILON = 1e-6
LEARNING_RATE = 0.01
TOLERANCE = 1e-6


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def summed_loss(x, y, weights, biases):
    """Return the summed squared error that the analytical gradients target."""
    activation = x
    for layer_weights, layer_bias in zip(weights, biases):
        activation = sigmoid(activation @ layer_weights + layer_bias)
    return 0.5 * np.sum((activation - y) ** 2)


def numerical_gradient(x, y, weights, biases, parameter):
    """Estimate a parameter's gradient with central differences."""
    gradient = np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        original = parameter[index]

        parameter[index] = original + EPSILON
        loss_plus = summed_loss(x, y, weights, biases)

        parameter[index] = original - EPSILON
        loss_minus = summed_loss(x, y, weights, biases)

        parameter[index] = original
        gradient[index] = (loss_plus - loss_minus) / (2 * EPSILON)
    return gradient


def main():
    np.random.seed(0)
    x = np.random.rand(3, 4)
    y = np.eye(3, 2)

    layers = np.array([layer(x, y, 5), layer(x, y, 5), layer(x, y, 2)])

    # The first pass creates the weights. Its update is then discarded below.
    layers[0].forwarding(layers)
    weights = [item.weights.copy() for item in layers]
    biases = [item.bias.copy() for item in layers]

    # Each update equals -learning_rate * gradient, so the gradient can be recovered.
    for item, layer_weights, layer_bias in zip(layers, weights, biases):
        item.weights = layer_weights.copy()
        item.bias = layer_bias.copy()
    layers[0].forwarding(layers)

    largest_difference = 0.0
    for number, item in enumerate(layers):
        checks = [
            ("weights", weights[number], item.weights),
            ("bias", biases[number], item.bias),
        ]
        for name, before, after in checks:
            analytical = (before - after) / LEARNING_RATE
            numerical = numerical_gradient(x, y, weights, biases, before)
            difference = np.abs(analytical - numerical).max()
            largest_difference = max(largest_difference, difference)
            print(f"Layer {number + 1} {name}: max difference = {difference:.2e}")

    if largest_difference > TOLERANCE:
        print("FAILED: backpropagation does not match the numerical gradients.")
        return 1

    print("PASSED: backpropagation matches the numerical gradients.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
