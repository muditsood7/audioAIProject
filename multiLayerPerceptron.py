import numpy as np
from random import random

# Save the activations and derivatives

# Implement back propagation

# Implement gradient descent

# Implement training method

# Train network with dummy data

# Make predictions

# Create class for Multi-Layered Perceptron
class MLP:

    # Constructor
    def __init__(self, num_inputs=3, num_hidden=[3, 5], num_outputs=2):

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [num_outputs]

        # Initiate random weights
        self.weights = []
        for i in range(len(layers) - 1):
            # Creates random arrays for each layer
            w = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(w)

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives

    # Create method for forward propagation
    def forward_propagate(self, inputs):

        activations = inputs
        self.activations[0] = inputs

        activations = inputs
        for i, w in enumerate(self.weights):
            # Calculate net inputs for a given layer
            net_inputs = np.dot(activations, w)

            # Calculate next activations
            activations = self.sigmoid(net_inputs)
            self.activations[i + 1] = activations

        return activations

    def back_propagate(self, error, verbose=False):

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], - 1)

            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))
        return error

    def gradient_descent(self, learning_rate):

        for i in range(len(self.weights)):
            weights = self.weights[i]
            # print("Original W{} {}".format(i, weights))

            derivatives = self.derivatives[i]

            weights += derivatives * learning_rate
            # print("Updated W{} {}".format(i, weights))

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):

                # Forward propagation
                output = self.forward_propagate(input)

                # Calculate error
                error = target - output

                # Back propagation
                self.back_propagate(error)

                # Gradient descent application
                self.gradient_descent(learning_rate=1)

                sum_error += self._mse(target, output)

            # report error
            print("Error: {} at epoch {}".format(sum_error/len(inputs), i))

    def _mse(self, target, output):
        return np.average((target - output)**2)

    def _sigmoid_derivative(self, x):
        return x * (1 - x)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':

    # Initialize mlp
    mlp = MLP(2, [5], 1)

    # Create dataset
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    # Train the mlp
    mlp.train(inputs, targets, 50, 0.1)

    # Create dummy data
    input = np.array([0.7, 0.2])
    target = np.array([0.9])

    output = mlp.forward_propagate(input)

    print("The network input is {}".format(input))
    print("The network output is {}".format(output))
