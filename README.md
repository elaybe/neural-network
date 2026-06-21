# Mathematically Rigorous MLP Forward Pass from Scratch (NumPy)

A highly optimized, dynamically initialized Multi-Layer Perceptron (MLP) forward propagation architecture built entirely from scratch using NumPy. 

This implementation is the final milestone achieved after **5 pages of manual mathematical derivation, rigorous matrix dimension tracking, and linear algebra verification** to ensure seamless batch processing without broadcasting anomalies.

---

## 1. Linear Algebra & Matrix Multiplication Compliance

To process a batch of inputs simultaneously without nested loops, the network strictly complies with the fundamental law of matrix multiplication: 

$$\text{Given } A \in \mathbb{R}^{m \times n} \text{ and } B \in \mathbb{R}^{p \times q}, \text{ the product } A \cdot B \text{ is defined if and only if } n = p.$$

The resulting matrix will strictly possess the dimensions:
$$\text{Output} \in \mathbb{R}^{m \times q}$$

---

## 2. Layer-by-Layer Architectural Mechanics

### Input Matrix ($X$)
The network accepts a dynamic batch of data. For a batch of $m$ samples, each having $d$ features:
* **Dimension:** $X \in \mathbb{R}^{m \times d}$
* *Example in code:* 4 samples, 2 features $\rightarrow X \in \mathbb{R}^{4 \times 2}$

---

### Hidden Layer (Dynamic Dimensioning)
#### 1. Node Determination Logic
The number of hidden neurons ($n_{\text{dim}}$) is calculated dynamically relative to the input dimensionality to scale the network’s capacity to handle non-linear combinations of features:
$$n_{\text{dim}} = 2^{d}$$
For $d = 2$ features, $n_{\text{dim}} = 2^2 = 4$ hidden neurons.

#### 2. Weight and Bias Shape Matching
To transform input features into the hidden space, the weight matrix dimensions must satisfy:
$$W^{[1]} \in \mathbb{R}^{d \times n_{\text{dim}}} \quad \rightarrow \quad W^{[1]} \in \mathbb{R}^{2 \times 4}$$
The bias vector is initialized as a row vector to facilitate proper broadcasting across the batch:
$$b^{[1]} \in \mathbb{R}^{1 \times n_{\text{dim}}} \quad \rightarrow \quad b^{[1]} \in \mathbb{R}^{1 \times 4}$$

#### 3. Linear Combination & Activation Equations
The pre-activation matrix $Z^{[1]}$ is computed via dot product:
$$Z^{[1]} = X \cdot W^{[1]} + b^{[1]}$$
$$\text{Dimensions Check:} \quad [m \times d] \cdot [d \times n_{\text{dim}}] = [m \times n_{\text{dim}}] \quad \rightarrow \quad [4 \times 2] \cdot [2 \times 4] = [4 \times 4]$$

The activation matrix $A^{[1]}$ applies the element-wise Logistic Sigmoid function:
$$A^{[1]} = \sigma(Z^{[1]}) = \frac{1}{1 + e^{-Z^{[1]}}}$$
$$\text{Dimension:} \quad A^{[1]} \in \mathbb{R}^{4 \times 4}$$

---

### Output Layer (Single Neuron)
#### 1. Weight and Bias Shape Matching
The output layer maps the 4 hidden representation features back to a single binary probability prediction ($n_{\text{out}} = 1$).
* **Weights:** $W^{[2]} \in \mathbb{R}^{n_{\text{dim}} \times 1} \quad \rightarrow \quad W^{[2]} \in \mathbb{R}^{4 \times 1}$
* **Bias:** $b^{[2]} \in \mathbb{R}^{1 \times 1}$

#### 2. Linear Combination & Activation Equations
The activation matrix from the hidden layer ($A^{[1]}$) acts as the input to this layer:
$$Z^{[2]} = A^{[1]} \cdot W^{[2]} + b^{[2]}$$
$$\text{Dimensions Check:} \quad [m \times n_{\text{dim}}] \cdot [n_{\text{dim}} \times 1] = [m \times 1] \quad \rightarrow \quad [4 \times 4] \cdot [4 \times 1] = [4 \times 1]$$

The final probability output $A^{[2]}$ is mapped between $0$ and $1$ via:
$$A^{[2]} = \sigma(Z^{[2]}) = \frac{1}{1 + e^{-Z^{[2]}}}$$
$$\text{Final Output Dimension:} \quad A^{[2]} \in \mathbb{R}^{4 \times 1}$$

---

## 3. Dynamic Weight Initialization (Modified Xavier/Glorot)
To avoid vanishing or exploding gradients during future training phases, weights are dynamically drawn from a normal distribution scaled by the input and output connectivity of that specific layer:

$$\sigma_{\text{init}} = \sqrt{\frac{2}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}}$$

```python
sigma = np.sqrt(2 / (w_dim + n_dim))
self.weights = np.random.normal(loc=0, scale=sigma, size=[w_dim, n_dim])
