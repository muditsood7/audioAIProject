import numpy as np

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
            a = np.zeroes(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeroes(layers[i], layers[i+1])
            derivatives.append(d)
        self.derivatives = derivatives

    # Create method for forward propagation
    def forward_propagate(self, inputs):

        activations = inputs
        self.activations[0] = inputs

        activations = inputs
        for i, w in self.weights:
            # Calculate net inputs for a given layer
            net_inputs = np.dot(activations, w)

            # Calculate next activations
            activations = self.sigmoid(net_inputs)
            self.activations[i + 1] = activations

        return activations

    def back_propagate(self, error):

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_derivatives_reshaped = current_activations.reshape(current_activations.shape[0], - 1)

            self.derivatives[i] = np.dot(current_activations, delta)


    def _sigmoid_derivative(self, x):
        return x * (1 - x)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # Initialize mlp
    mlp = MLP()

    # Create inputs
    inputs = np.random.rand(mlp.num_inputs)

    # Outputs
    outputs = mlp.forward_propagate(inputs)

    print("The network input is {}".format(inputs))
    print("The network output is {}".format(outputs))
