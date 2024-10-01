
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            if isinstance(v, Value):
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
        dot.node(uid, label="{ data %.4f}" % (n.data, ), shape='record')
        if n._op:
            dot.node(name = uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot


import math
import random

class Value:
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
        self._backward = _backward
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
        self._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self, other), f'**{other}')

        def _backward():
            self.grad += other*self.data**(other-1) * out.grad
        self._backward = _backward

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
        self._backward = _backward
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

        self._backward = _backward

        return out

    def backprop(self):
        # Topo sort
        topo = []
        visited = set()
        def build_topo(v):
            print(f'Node: {v}, type: {type(v)}')
            if isinstance(v, Value) and v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1

        # def build_topo(v):
        #     if isinstance(v, Value):
        #         if v not in visited:
        #             visited.add(v)
        #             for child in v._prev:
        #                 build_topo(child)
        #             topo.append(v)
            # else:
            #     print(f'Node {v} has been skiped from build_topo, as has type {type(v)}')
        
        print(f'Topo: {topo}')
        for self in reversed(topo):
            self._backward()

        # for self in reversed(topo):
        #     print(f'Label: {self._label}, value: {self.data}, grad: {self.grad}')

    def zero_grad(self):
        if not self._prev:
            return 
        
        self.grad = 0
        for node in self._prev:
            try:
                node.zero_grad()
            except:
                pass
                # print(f'Node {node}, with type {type(node)} is not a Value')

class Perceptron():
    def __init__(self, n):
        self.w = []
        for _ in range(n):
            self.w.append(Value(random.uniform(-1,1))) 

        self.b = Value(random.uniform(-1,1))

    def __call__(self, input):
        input = [Value(input[i]) for i in range(len(input))]
        s = self.b # Add the bias
        for w, x in zip(self.w, input):
            s += w*x
        f = s.tanh() # Apply relu non linearity

        return f

    def parameters(self):
        return self.w + [self.b]
    


if __name__ == '__main__':

    a = Value(2)
    b = Value(3)
    c = Value(4)
    d = a*b + c

    draw_dot(d)
    """per = Perceptron(2)
    # AND gate data
    X = [[0.,0.],
        [0.,1.],
        [1.,0.],
        [1.,1.]]
    y = [0,0,1,1]


    for i in range(1):
        x_sample, y_sample = X[i%4], y[i%4]
        # print(f'Input: {x_sample}, Target: {y_sample}')
        pred = per(x_sample)
        L = (pred - y_sample)**2
        L.zero_grad()
        print(type(L))
        draw_dot(L)
        # L.grad = 1
        L.backprop()

        pars = per.parameters()
        for pa in pars:
            print(pa.grad)
        for p in pars:
            p.data += -0.00001*p.grad
        
        

        # print(L.data)
        # print(per.w)

    X = [[0.,0.],
        [0.,1.],
        [1.,0.],
        [1.,1.]]
    y = [0,0,1,1]

    for sample in X:
        print(per.forward(sample))
    print(y)"""


