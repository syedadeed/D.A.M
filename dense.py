import numpy as np

class Dense:
    def __init__(self, input_size, output_size, weight=None, bias=None):
        self.weight = weight if weight is not None else np.random.randn(output_size, input_size)
        self.bias = bias if bias is not None else np.random.randn(output_size, 1)

    def forward_propogation(self, input_value):
        self.input_value = input_value
        return np.dot(self.weight, self.input_value) + self.bias

    def backward_propogation(self, output_gradient, learning_rate=0.01):
        weight_gradient = np.dot(output_gradient, self.input_value.T)
        bias_gradient = output_gradient
        input_gradient = np.dot(self.weight.T, output_gradient)

        self.weight -= learning_rate * weight_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient
