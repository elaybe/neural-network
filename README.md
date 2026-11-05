# Neural Network from Scratch

A fully connected neural network (multilayer perceptron) implemented in NumPy, with
forward propagation and backpropagation derived and written by hand.

The goal of this project is to understand how backpropagation works at the matrix level.
It is an educational implementation, not a production library.

## Features

- Vectorized forward and backward passes
- Configurable number of layers and neurons
- Weight dimensions inferred from the input at the first forward pass
- Xavier/Glorot normal weight initialization
- Sigmoid activation in all layers
- Full-batch gradient descent

## Quick Start

Requires Python 3 and NumPy.

```bash
pip install numpy
python NeuralNetwork.py
```

The script trains the network on a small example and prints the loss every 100 epochs.

## Default Network

| Layer | Neurons | Activation |
|---|---|---|
| Hidden 1 | 30 | Sigmoid |
| Hidden 2 | 30 | Sigmoid |
| Hidden 3 | 30 | Sigmoid |
| Output | 10 | Sigmoid |

The input has 3 samples with 4 features each. The learning rate is 0.01.

## How It Works

### Forward Propagation

For layer $L$, with $A^{(0)} = X$:

$$Z^{(L)} = A^{(L-1)} W^{(L)} + b^{(L)}$$

$$A^{(L)} = \sigma(Z^{(L)}) = \frac{1}{1 + e^{-Z^{(L)}}}$$

### Loss

The gradients are computed for the summed squared error:

$$L = \frac{1}{2} \sum_{i,j} \left(A^{(out)}_{ij} - Y_{ij}\right)^2$$

The value printed during training is the mean version of this loss, used for monitoring only.

### Backpropagation

Define the gradient of the loss with respect to each layer's pre-activation as:

$$\delta^{(L)} = \frac{\partial L}{\partial Z^{(L)}}$$

For the output layer:

$$\delta^{(out)} = (A^{(out)} - Y) \odot A^{(out)} \odot (1 - A^{(out)})$$

For a hidden layer:

$$\delta^{(L)} = \left(\delta^{(L+1)} (W^{(L+1)})^T\right) \odot A^{(L)} \odot (1 - A^{(L)})$$

The parameter gradients are:

$$\frac{\partial L}{\partial W^{(L)}} = (A^{(L-1)})^T \delta^{(L)}$$

$$\frac{\partial L}{\partial b^{(L)}} = \sum_{i} \delta^{(L)}_{i}$$

### Parameter Update

$$W^{(L)} \leftarrow W^{(L)} - \eta \frac{\partial L}{\partial W^{(L)}}$$

$$b^{(L)} \leftarrow b^{(L)} - \eta \frac{\partial L}{\partial b^{(L)}}$$

### Weight Initialization

$$W^{(L)} \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

Bias values are initialized to zero.

## Verification

- The analytical gradients of all layers' weights and biases are checked against numerical
  gradients on a 3-layer network, with a maximum absolute difference of about `1e-10`.
  Run the check with: `python test_gradients.py`
- In one example run, the printed loss fell from about `0.12` to about `0.00005`
  after 100,000 epochs.

The example has only 3 samples, so this confirms that the computation works.
It does not measure generalization.

## Limitations

- Sigmoid is the only activation function.
- The only training method is full-batch gradient descent.
- There is no train/test split or real-dataset evaluation.
- No random seed is set, so results vary between runs.
