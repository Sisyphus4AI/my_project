import numpy as np

# A simple Python program
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    name = input("Enter your name: ")
    # Example of a simple neural network using numpy

    # Input data
    inputs = np.array([1, 2, 3])
    # Weights
    weights = np.array([0.1, 0.2, 0.3])
    # Bias
    bias = 0.5

    # Simple forward pass
    output = np.dot(inputs, weights) + bias
    print(f"Neural Network Output: {output}")