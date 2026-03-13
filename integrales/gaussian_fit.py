import math
import random
import pandas as pd
import numpy as np
import sys


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
            print("log: ", math.log(self.data+0.000001))
            other.grad += self.data**other.data*math.log(self.data+1) #* out.grad
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

class Perceptron():
    def __init__(self, n):
        self.w = []
        for _ in range(n):
            self.w.append(Value(random.uniform(-1,1))) 
        self.b = Value(random.uniform(-1,1))

    def __call__(self, input):
        input = [Value(input[i]) for i in range(len(input))]
        s = Value(0)
        
        for w, x in zip(self.w, input):
            s += w*x
        s += self.b # Add the bias
        f = s.tanh() # Apply relu non linearity
        return f

    def parameters(self):
        return self.w + [self.b]
    
class Gaussian():
    def __init__(self):
        self.mu = Value(random.uniform(-1,1), label="mu")
        self.de = Value(random.uniform(-1,1), label="de") # Standard deviation
    
    def __call__(self, x):
        x = Value(x, label="x")
        one = Value(1, label="one")
        pi = Value(3.141592653589793, label="pi")
        e = Value(2.718281828459045, label="e")
        res = (one/(self.de*(Value(2)*pi)**0.5))*e**(-Value(0.5)*((x-self.mu)/self.de)**2)
        # (1/(de*np.sqrt(2*np.pi)))*np.e**(-1/2*((x-mu)/de)**2)
        return res
    
    def parameters(self):
        return [self.mu, self.de]
    

from graphviz import Digraph

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
    for _ in range(epochs):
        for x_sample, y_sample in zip(X_train, y_train):
            pred = model(x_sample)
            print("DATA: ", pred, y_sample, pred-y_sample, (pred - y_sample)**2.0)
            L = (pred - y_sample)**2.0
            dot = draw_dot(L)
            dot.render(filename='L', format='png', cleanup=True)
            # print("L.data: ", L.data)
            # sys.exit()
            L.zero_grad()
            pars = model.parameters()
            for p in pars:
                print("gradient_1: ", p.grad)
            L.backward()

            pars = model.parameters()
            for p in pars:
                p.data += -lr*p.grad
                print("gradient_2: ", p.grad)
            print()
            loss_values.append(L.data)
        # print(sum(loss_values)/len(loss_values))
        # print("LOSS VALUES: ", loss_values)
        loss_values = []
    print("PARAMS: ", model.parameters())

def evaluate_model(model, X_test, y_test):
    for x_sample, y_sample in zip(X_test, y_test):
        res = model(x_sample)
        print(f'Input: {x_sample}, Target: {y_sample}, Prediction: {res.data}')

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

if __name__ == '__main__':
    random.seed(42)
    # Model initialization
    # per = Perceptron(2)
    gau = Gaussian()

    res = gau(5)
    dot = draw_dot(res)
    dot.render(filename='my_graph', format='png', cleanup=True)

    X, y = get_data()

    # Training the model
    train_model(gau, X, y, lr=1, epochs=2)

    # Evaluating the model
    # evaluate_model(per, X, y)

    



