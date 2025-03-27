import numpy as np

"""
N -> Num samples
K -> Num classes
D -> Dimensionality of input X

X -> (N,D)
W -> (D,K)
b -> (K,)
"""

class FFN:
    def __init__(self,K,D): # K is the # of classes, and D the input dimension
        self.W = np.random.rand(D,K)
        self.b = np.random.rand(K)
    
    def forward(self, X):
        preds = np.dot(X,self.W) + self.b # Gives (N,K)
        scores = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True) # Softmax
        return scores

    def compute_loss(self, preds, Y, N): # preds -> (N,K) and Y -> (N)
        return sum(-np.log(preds[np.arange(N),Y])) / N # Cross entropy loss

    def backward(self, preds, Y, X, N):
        # Compute score gradients
        dscores = preds.copy()
        dscores[np.arange(N),Y] -= 1
        dscores /= N # (N,K)

        dW = np.dot(X.T, dscores) # X.T is (D,N) and dscores (N,K)
        db = np.sum(dscores, axis=0, keepdims=True)

        return dW, db

# Hiperparameters
iterations = 10000
lr = 0.01
N = 4
K = 2
D = 2

# Data
X = np.array([[0,1],
              [1,0],
              [1,1],
              [0,0]])
Y = np.array([0,0,1,1]) # Indexes of the correct classes

# Model declaration
model = FFN(K,D)

# Training loop
for _ in range(iterations): 
    # Forward pass
    preds = model.forward(X)
    print(model.compute_loss(preds, Y, N))
    
    # Compute gradients
    dW, db = model.backward(preds, Y, X, N)

    # Update network
    model.W += -lr*dW
    model.b += -lr*db.squeeze()

# Testing
X = np.array([[0,1],
              [1,0],
              [1,1],
              [0,0]])
print(model.forward(X))
