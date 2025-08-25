import math

# Vanilla softmax implementation
# This implementation is not numerically stable and can lead to overflow issues
# when the input values are large.
def vanilla_softmax(x):
    probs = [math.exp(p) for p in x]
    s = sum(probs)
    return [p/s for p in probs]

# Online softmax implementation, which updates the probabilities incrementally
# This is a naive implementation that does not handle numerical stability issues.
def unsafe_online_softmax(x): 
    denom = 0
    probs = []

    for s in x:
        new_denom = denom + math.exp(s)
        s = math.exp(s) / new_denom
        probs = [p * denom / new_denom for p in probs]
        denom = new_denom
        probs.append(s)
    return probs

# Stable online softmax implementation
# This implementation uses a numerically stable approach to avoid overflow issues
# by subtracting the maximum value from each input before exponentiation.
def stable_online_softmax(x):
    denom = 0
    m = -math.inf
    probs = []

    for s in x:
        new_m = max(m, s)
        new_denom = denom*math.exp(m-new_m) + math.exp(s-new_m)
        s = math.exp(s-new_m) / new_denom if new_denom else 0.0
        probs = [p * denom*math.exp(m-new_m) / new_denom for p in probs]
        denom = new_denom
        m = new_m
        probs.append(s)
    return probs

if __name__ == "__main__":
    # Example usage
    X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    soft = stable_online_softmax(X)
    v_soft = vanilla_softmax(X)
    print(sum(soft), soft)
    print(sum(v_soft), v_soft) # To check correct stable_online results