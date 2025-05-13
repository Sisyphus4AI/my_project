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

    print(greet(name))
# This is a simple test script to demonstrate the use of numpy and a greeting function.
# It includes a simple neural network forward pass and a greeting function. 
# The script is designed to be run as a standalone program.