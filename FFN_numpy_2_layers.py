import numpy as np

"""
N -> Num samples
K -> Num classes
D -> Dimensionality of input X
"""

class FFN:
    def __init__(self,K,D,H): # K is the # of classes, and D the input dimension
        self.W1 = np.random.rand(D,H)
        self.W2 = np.random.rand(H,K)
        self.b1 = np.zeros(H) # np.random.rand(H) does not work
        self.b2 = np.zeros(K) # np.random.rand(K) 
        self.hidden = np.zeros((N,H))
    
    def forward(self, X):
        # First layer computation with ReLU
        self.hidden = np.maximum(0, np.dot(X,self.W1) + self.b1) # X is (N,D), W1 is (D,H). Gives (N,H)
        # Score computation
        scores = np.dot(self.hidden, self.W2) + self.b2 # Gives (N,K)
        # Apply softmax
        # scores -= np.max(scores, axis=1, keepdims=True)  # Normalize scores to prevent large exponents
        scores = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        return scores

    def compute_loss(self, preds, Y, N): # preds -> (N,K) and Y -> (N)
        return sum(-np.log(preds[np.arange(N),Y])) / N # Cross entropy loss

    def backward(self, preds, Y, X, N):
        # Compute score gradients
        dscores = preds.copy()
        dscores[np.arange(N),Y] -= 1
        dscores /= N # (N,K)

        # Compute gradients of the last layer
        dW2 = np.dot(self.hidden.T, dscores) # hidden is (N,H) and dscores (N,K). Gives (H,K)
        db2 = np.sum(dscores, axis=0, keepdims=True)

        # Compute gradients of the first layer with the RELU
        dhidden = np.dot(dscores, self.W2.T) # (N,K) and (K,H) gives (N,H)
        dhidden[self.hidden == 0] = 0 # Also dhidden[self.hidden == 0] = 0??

        dW1 = np.dot(X.T, dhidden) # X.T is (D,N) and dhidden (N,H). Gives (D,H)
        db1 = np.sum(dhidden, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

# Hiperparameters
iterations = 1000000
lr = 0.01
N = 4
K = 2
D = 2
H = 10

# Data
X = np.array([[0,1],
              [1,0],
              [1,1],
              [0,0]])
Y = np.array([1,1,0,0]) # Indexes of the correct classes

# Model declaration
model = FFN(K,D,H)

# Training loop
for i in range(iterations): 
    # Forward pass
    preds = model.forward(X)
    
    
    # Compute gradients
    dW1, db1, dW2, db2 = model.backward(preds, Y, X, N)

    # Update network
    model.W1 += -lr*dW1
    model.b1 += -lr*db1.squeeze()

    model.W2 += -lr*dW2
    model.b2 += -lr*db2.squeeze()

    if i%10000 == 0:
        print(model.compute_loss(preds, Y, N))

# Testing
X = np.array([[0,1],
              [1,0],
              [1,1],
              [0,0]])
print(model.forward(X))
