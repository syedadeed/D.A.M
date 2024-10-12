import numpy as np

class Activation:
    def __init__(self, activation_function, activation_function_prime):
        self.activation_function = activation_function
        self.activation_function_prime = activation_function_prime

    def forward_propogation(self, input_value):
        self.input_value = input_value
        return self.activation_function(input_value)

    def backward_propogation(self, output_gradient):
        return np.multiply(output_gradient, self.activation_function_prime(self.input_value))

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(input_value):
            input_value = np.clip(input_value, -709, 709)
            return 1 / (1 + np.exp(-input_value))

        def sigmoid_prime(input_value):
            s = sigmoid(input_value)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class Relu(Activation):
    def __init__(self):
        def relu(input_value):
            return np.where(input_value < 0, 0, input_value)

        def relu_prime(input_value):
            return np.where(input_value < 0, 0, 1)

        super().__init__(relu, relu_prime)
