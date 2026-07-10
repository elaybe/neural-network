# Mathematically Rigorous MLP from Scratch (NumPy)

A highly optimized, dynamically initialized Multi-Layer Perceptron (MLP) architecture featuring complete forward and backward propagation, built entirely from scratch using NumPy. 

This implementation represents a rigorous translation of matrix calculus into computational code. It features dynamic weight dimensioning, a vectorized structural loop for arbitrary layer depths, and strict compliance with the chain rule for matrix gradients to avoid broadcasting anomalies during batch updates.

---

## 1. Linear Algebra & Matrix Multiplication Compliance

To process an entire batch of inputs simultaneously without nested loops, the network strictly complies with the fundamental laws of matrix calculus:
Given $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{p \times q}$, the product $A \cdot B$ is defined if and only if $n = p$. The resulting matrix dimensions are strictly: $\text{Output} \in \mathbb{R}^{m \times q}$.

---

## 2. Forward Propagation Mechanics

### Input & Hidden Representation
The network accepts a dynamic batch of data $X \in \mathbb{R}^{m \times d}$ where $m$ is the batch size and $d$ is the number of input features. For any given layer $L$ with $N$ neurons, receiving inputs from a preceding layer with $K$ features:

* **Weight Matrix ($W^{(L)}$):** Maps the input features to the neuron space. 
  $$W^{(L)} \in \mathbb{R}^{K \times N}$$
* **Bias Vector ($b^{(L)}$):** Formatted as a row vector to facilitate proper row-wise broadcasting across the batch.
  $$b^{(L)} \in \mathbb{R}^{1 \times N}$$

### Layer-by-Layer Equations (Vectorized)
For each layer, the pre-activation matrix $Z^{(L)}$ and the post-activation matrix $A^{(L)}$ are evaluated via the element-wise Logistic Sigmoid function:

$$Z^{(L)} = A^{(L-1)} \cdot W^{(L)} + b^{(L)}$$
$$A^{(L)} = \sigma(Z^{(L)}) = \frac{1}{1 + e^{-Z^{(L)}}}$$

### Dimensions Check for the Configured 4-Layer Network
Given $m = 3$ samples, $d = 4$ features, 3 hidden layers with $30$ neurons each, and $10$ output neurons:
* **Layer 1 (Hidden):** $[3 \times 4] \cdot [4 \times 30] + [1 \times 30] = [3 \times 30] \rightarrow A^{(1)} \in \mathbb{R}^{3 \times 30}$
* **Layer 2 (Hidden):** $[3 \times 30] \cdot [30 \times 30] + [1 \times 30] = [3 \times 30] \rightarrow A^{(2)} \in \mathbb{R}^{3 \times 30}$
* **Layer 3 (Hidden):** $[3 \times 30] \cdot [30 \times 30] + [1 \times 30] = [3 \times 30] \rightarrow A^{(3)} \in \mathbb{R}^{3 \times 30}$
* **Layer 4 (Output):** $[3 \times 30] \cdot [30 \times 10] + [1 \times 10] = [3 \times 10] \rightarrow A^{(4)} \in \mathbb{R}^{3 \times 10}$

---

## 3. Backpropagation & Matrix Calculus Derivation

The backward pass evaluates the exact analytical gradients of the Mean Squared Error (MSE) loss function with respect to all trainable parameters by traversing the layers in reverse order (`reversed(layers)`).

#### Loss Function
$$L = \frac{1}{2m} \sum_{i=1}^{m} (A^{(out)} - Y)^2$$

#### Full Chain Rule Expansion for Deep Hidden Layers
As visualized in the architectural mathematical derivation, the gradient of the loss with respect to the weights of a deep hidden layer ($W^{(1)}$) expands fully through every single transition vector via element-wise and matrix dot products:

$$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial \hat{a}} \odot \frac{\partial \hat{a}}{\partial Z^{(4)}} \cdot \frac{\partial Z^{(4)}}{\partial A^{(3)}} \odot \frac{\partial A^{(3)}}{\partial Z^{(3)}} \cdot \frac{\partial Z^{(3)}}{\partial A^{(2)}} \odot \frac{\partial A^{(2)}}{\partial Z^{(2)}} \cdot \frac{\partial Z^{(2)}}{\partial A^{(1)}} \odot \frac{\partial A^{(1)}}{\partial Z^{(1)}} \cdot \frac{\partial Z^{(1)}}{\partial W^{(1)}}$$

#### Output Layer Gradients ($L = \text{Output}$)
Applying the chain rule to compute the error gradient relative to the linear combination $Z^{(out)}$:
$$\delta^{(out)} = \frac{\partial L}{\partial Z^{(out)}} = \frac{\partial L}{\partial A^{(out)}} \odot \sigma'(Z^{(out)})$$
$$\delta^{(out)} = (A^{(out)} - Y) \odot \left( A^{(out)} \odot (1 - A^{(out)}) \right)$$

#### Hidden Layer Gradients & Error Propagation
For any hidden layer, the error gradient $\delta^{(L)}$ depends on the accumulated gradient from the subsequent layer $\delta^{(L+1)}$ and the **original** un-updated weights $W^{(L+1)}$:

$$\delta^{(L)} = \frac{\partial L}{\partial Z^{(L)}} = \left( \delta^{(L+1)} \cdot (W^{(L+1)})^T \right) \odot \left( A^{(L)} \odot (1 - A^{(L)}) \right)$$

#### Parameter Gradient Evaluation & Optimization Updates
Once $\delta^{(L)}$ is explicitly isolated for any layer, the weight and bias gradients are evaluated via matrix transposition to match parameter dimensions exactly, preventing premature weight corruption by storing state variables locally:

$$\frac{\partial L}{\partial W^{(L)}} = (A^{(L-1)})^T \cdot \delta^{(L)}$$
$$\frac{\partial L}{\partial b^{(L)}} = \sum_{\text{rows}} \delta^{(L)}$$

The optimization step updates the parameters via Gradient Descent given a learning rate $\eta$:
$$W^{(L)} \leftarrow W^{(L)} - \eta \cdot \frac{\partial L}{\partial W^{(L)}}$$
$$b^{(L)} \leftarrow b^{(L)} - \eta \cdot \frac{\partial L}{\partial b^{(L)}}$$

---

## 4. Normalized Xavier/Glorot Initialization

To prevent vanishing or exploding gradients across deep configurations, weights are drawn from a normal distribution scaled dynamically by the input connectivity ($\text{fan}_{\text{in}}$) and output connectivity ($\text{fan}_{\text{out}}$) of each specific layer:

$$\sigma_{\text{init}} = \sqrt{\frac{2}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}}$$

```python
sigma = np.sqrt(2 / (col + self.neurons))
self.weights = np.random.normal(loc=0, scale=sigma, size=[col, self.neurons])
self.bias = np.zeros((1, self.neurons))
