import torch

torch.manual_seed(42)

# Model definition
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1

W1 = torch.randn(input_size,hidden_size, requires_grad=True) # Input layer
b1 = torch.randn(hidden_size, requires_grad=True) # Bias 1

W2 = torch.randn(hidden_size,output_size, requires_grad=True) # Output layer
b2 = torch.randn(output_size, requires_grad=True) # Bias 2

# Non-linearity
def relu(x):
    return torch.max(x, torch.zeros_like(x))

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Dataset
X = torch.tensor([[0.,0.], [0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([1.,0.,0.,1.])

# Optimizer
optimizer = torch.optim.SGD([W1,b1,W2,b2], lr=learning_rate)

# Training loop
for i in range(100000):
    # Forward pass
    hidden_layer = sigmoid(X[i%4] @ W1 + b1)
    res = sigmoid(hidden_layer @ W2 + b2)

    loss = (res - y[i%4]) ** 2 # Squared error
    
    # Compute gradients
    optimizer.zero_grad()
    loss.backward()

    # Update network
    optimizer.step()

    if i%100 == 0:
        print("Current loss: ", loss)


# Testing final result
for i in range(4):
    hidden_layer = sigmoid(X[i%4] @ W1 + b1)
    res = sigmoid(hidden_layer @ W2 + b2)
    print(X[i], " -> ", res)
