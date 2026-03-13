import math
import random
import pandas as pd
import numpy as np
import sys
from graphviz import Digraph
import matplotlib.pyplot as plt


class Value:
    """ Value is a class that represents a value in the computation graph of the neural network.
    It has a data attribute that stores the value of the node, a grad attribute that stores the gradient,
    a _prev attribute that stores the children of the node, a _op attribute that stores the operation that
    the node is performing and a _backward attribute that stores the backward function of the node.
    """
    def __init__(self, data, _children=(), _op='', label = ''):
        self.data = data
        self._prev = _children
        self.grad = 0.0
        self._op = _op
        self._label = label
        self._backward = lambda: None

    def __repr__(self):
        return f'Value: {self.data}'
    
    def __add__(self, other):
        self = self if isinstance(self, Value) else Value(self)
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad

            other.grad += out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        self = self if isinstance(self,Value) else Value(self)
        return self + (-other)

    def __mul__(self, other):
        self = self if isinstance(self, Value) else Value(self)
        other = other if isinstance(other, Value) else Value(other)

        out =  Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        # assert isinstance(other, (int, float))
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data**other.data, (self, other), f'**{other.data}')

        def _backward():
            self.grad += other.data*self.data**(other.data-1) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        self = self if isinstance(self, Value) else Value(self)
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        self = self if isinstance(self, Value) else Value(self)

        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        # Topo sort
        topo = []
        visited = set()
        def build_topo(v):
            if not isinstance(v, Value):
                v = Value(v)
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1

        for self in reversed(topo):
            self._backward()

    def zero_grad(self):
        self.grad = 0
        for node in self._prev:
            node.zero_grad()
    
class Gaussian():
    def __init__(self):
        self.mu = Value(random.uniform(10, 30), label="mu")
        self.de = Value(random.uniform(1, 10), label="de") # Standard deviation
        self.a = Value(random.uniform(10, 30), label="a")
    
    def __call__(self, x):
        x = Value(x, label="x")
        e = -(x-self.mu)**2/(Value(2)*self.de**2)
        res = self.a * e.exp()
        return res
    
    def parameters(self):
        return [self.mu, self.de, self.a]


def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if not isinstance(v, Value):
            v = Value(v)
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        if not n._label:
            dot.node(uid, label="{ data %.4f}" % (n.data, ), shape='record')
        else:
            dot.node(uid, label=n._label, shape='record')
        if n._op:
            dot.node(name = uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot

def train_model(model, X_train, y_train, lr, epochs):
    loss_values = []
    print("PARAMS: ", model.parameters())
    for e in range(epochs):
        for x_sample, y_sample in zip(X_train, y_train):
            pred = model(x_sample)
            L = (pred - y_sample)**2.0
            dot = draw_dot(L)
            dot.render(filename='backprop_graph', format='png', cleanup=True)
            L.zero_grad()
            pars = model.parameters()
            L.backward()
            for p in pars:
                p.data += -lr*p.grad
            loss_values.append(L.data)
        print(f"EPOCH {e}: LOSS: {sum(loss_values)/len(loss_values)}")
        loss_values = []
    print("PARAMS: ", model.parameters())

def evaluate_model(model, X_test, y_test):
    mae = []
    for x_sample, y_sample in zip(X_test, y_test):
        res = model(x_sample)
        mae.append(abs(res.data - y_sample))
    print("MAE: ", sum(mae)/len(mae))

def get_data():
    data = pd.read_csv("data.csv")["texture_mean"]
    num_div = 100
    min = data.min()
    max = data.max()
    step_size = (max - min)/num_div
    # print(min, max)
    histo = np.array([0]*num_div)
    for el in data:
        index = int((el - min)/step_size)
        histo[index-1]+=1
    bin_edges = np.arange(min, max, step_size)
    return bin_edges, histo

def plot_data_and_fit(model, X, y):
    """Plots the actual data points and the calculated function curve."""        
    plt.figure(figsize=(10, 6))
    
    # Plot the actual data
    plt.scatter(X, y, color='blue', label='Data', alpha=0.6)
    
    # Generate smooth points for the calculated function
    x_min, x_max = min(X), max(X)
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1
        
    x_dense = np.linspace(x_min, x_max, 500)
    y_calc = [model(x).data for x in x_dense]
    
    plt.plot(x_dense, y_calc, color='red', label='Calculated Function', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data vs. Calculated Gaussian Fit')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('data_and_fit.png')
    print("Plot saved successfully to 'data_and_fit.png'.")
    plt.show()

if __name__ == '__main__':
    random.seed(42)
    # Model initialization
    gau = Gaussian()

    res = gau(5)

    X, y = get_data()

    # Training the model
    train_model(gau, X, y, lr=0.0001, epochs=100)

    # Evaluating the model
    evaluate_model(gau, X, y)
    
    # Plot the results
    plot_data_and_fit(gau, X, y)



