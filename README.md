# 🧠 AI Adventures 🚀🤖🔍

This repo contains my hands-on implementations of various AI algorithms, created to deepen my understanding of fundamental concepts. 💡 The code prioritizes clarity and minimizes external dependencies, making these implementations both educational and accessible.

## 📚 Projects

### 🔹 Perceptron ([perceptron.py](perceptron.py))

🧮 This project implements a simple perceptron inspired by Andrej Karpathy's [Micrograd](https://github.com/karpathy/micrograd/tree/master). 

✨ It features automatic gradient computation for various operations:
- ➕ Basic math: addition, subtraction, negation, multiplication, division, power
- 📈 Activation functions: tanh, ReLU
- 🔄 Other functions: exponential

🧩 The Perceptron class uses a tanh activation function and is trained with backpropagation on a dataset generated from the logical expression $AB + A\overline{B}$.

### 🔹 Gradient Descent ([gradient_descent.ipynb](gradient_descent.ipynb))

🎯 This project demonstrates optimization of two functions:
- 📊 Simple quadratic function: $$f(x) = x² + y²$$
- 🌊 Complex trigonometric function: $$f(x) = sin(1/2 * x^2 - 1/4 * y^2 + 3) * cos(2*x + 1 - e^y)$$

⚙️ The optimization uses gradient descent with manually specified partial derivatives for each term.

### 🔹 Feed-Forward Networks in NumPy ([FFN_numpy.py](FFN_numpy.py))

🔬 A pure NumPy implementation of feed-forward neural networks, demonstrating the fundamentals of forward and backward propagation without relying on deep learning frameworks.

### 🔹 Two-Layer Feed-Forward Network ([FFN_numpy_2_layers.py](FFN_numpy_2_layers.py))

🏗️ An extension of the NumPy implementation that specifically focuses on a two-layer architecture, providing a clear illustration of multi-layer perceptrons.

### 🔹 Feed-Forward Networks in PyTorch ([FFN_torch.py](FFN_torch.py))

🔥 A PyTorch implementation of feed-forward networks that demonstrates how to leverage a modern deep learning framework while maintaining an understanding of the underlying concepts.

### 🔹 Recurrent Neural Networks in PyTorch ([RNN_torch.py](RNN_torch.py))

⏱️ Implementation of recurrent neural networks using PyTorch, exploring sequence modeling and the handling of temporal data.

## 🚀 Getting Started

To run these projects, you'll need Python with NumPy and PyTorch installed:

```bash
pip install numpy torch matplotlib jupyter
```