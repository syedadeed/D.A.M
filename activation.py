import numpy as np

class Activation:
    def __init__(self, activation_function, derivative_activation_function):
        self.activation_function = activation_function
        self.derivative_activation_function = derivative_activation_function

    def forward_propogation(self, input_value):
        self.input_value = input_value
        return self.activation_function(input_value)

    def backward_propogation(self, output_gradient):
        return np.multiply(output_gradient, self.derivative_activation_function(self.input_value))

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)
