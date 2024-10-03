# Ai_adventures 🚀🤖🔍

This repo contains the code of some of my implementations of AI algorithms, which I use to deepen my understanding of the concepts. It tries to use the minimum external libraries as possible, and the code is written in a way that is easy to understand.

## Projects

### Perceptron

This project contains the implementation of a simple perceptron. It is based on the repo [Micrograd](https://github.com/karpathy/micrograd/tree/master) from Andrej Karpathy. 
It implements all the automatic computation of the gradients for any given architecture that uses the operations implemented (addition, subtraction, negation, multiplication, division, power, tanh, relu, and exponential).

Also, it defines the Perceptron class, which is a simple perceptron with a tanh activation function. It is trained using the backpropagation algorithm and the dataset generated from the logic expression $AB + A\overline{B}$.


### Gradient Descent

This project contains the optimization of two functions, $$f(x) =  x² + y²$$ and $$f(x)=sin(1/2 * x^2 - 1/4 * y^2 + 3) * cos(2*x + 1 - e^y)$$

The optimization is done using the gradient descent algorithm, where the gradients are computed by specifying directly the partial derivatives of each term of the function.
    



