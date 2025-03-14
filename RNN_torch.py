import torch

torch.manual_seed(42)

# Model definition
input_size = 26
hidden_size = 8
output_size = 26
learning_rate = 0.1

W1 = torch.randn(input_size,hidden_size, requires_grad=True) # Precesses input
b1 = torch.randn(hidden_size, requires_grad=True)

W2 = torch.randn(hidden_size,hidden_size, requires_grad=True) # Processes previos hidden state

W3 = torch.randn(hidden_size, output_size, requires_grad=True) # Output layer
b2 = torch.randn(output_size, requires_grad=True)

# Non-linearity
def relu(x):
    return torch.max(x, torch.zeros_like(x))

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Dataset
def one_hot_encoding(x):
    enc = torch.zeros(26)
    enc[ord(x) - ord('a')] = 1
    return enc

def one_hot_to_char(x):
    ind = torch.argmax(x).item()
    return chr(ord('a') + ind)

# Generate the one hot encoding of the alphabet
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
X = [one_hot_encoding(x) for x in alphabet]

# Optimizer
optimizer = torch.optim.SGD([W1,b1,W2,b2,W3], lr=learning_rate)

prevH = torch.zeros(hidden_size, requires_grad=True)

# Training loop
iterations = 100000
for i in range(iterations):
    # Forward pass
    hidden_layer = sigmoid(X[i%26]@W1 + (prevH)@W2 + b1)
    res = sigmoid(hidden_layer @ W3 + b2)

    loss = torch.nn.functional.binary_cross_entropy(res, X[(i+1)%26])
    
    # Compute gradients
    optimizer.zero_grad()
    loss.backward()

    # Update network3
    optimizer.step()

    prevH = hidden_layer#.detach()

    if i%10000 == 0:
        print(i/iterations*100, "% Current loss: ", loss)


# Generate characters
with torch.no_grad():
    prevH = torch.zeros(hidden_size)
    for i in range(26):
        hidden_layer = sigmoid(X[i%26]@W1 + (prevH)@W2 + b1)
        res = sigmoid(hidden_layer @ W3 + b2)

        prevH = hidden_layer.detach().clone()
        print(one_hot_to_char(X[i%26]), " -> ", one_hot_to_char(res))


