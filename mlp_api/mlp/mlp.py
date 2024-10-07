import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializa as dimensões da MLP
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicializa os pesos da camada oculta e da camada de saída
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # Inicializa os bias
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))


    def train(self, input_data, output_label, epochs=1000, learning_rate=0.1):
        for _ in range(epochs):
            # Forward pass
            hidden_layer_input = np.dot(input_data, self.weights_input_hidden)
            hidden_layer_output = self.activation_function(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
            output_layer_output = self.activation_function(output_layer_input)

            # Calcula o erro
            error = output_label - output_layer_output
            
            # Backpropagation
            output_gradient = error * self.activation_derivative(output_layer_output)
            hidden_error = np.dot(output_gradient, self.weights_hidden_output.T)
            hidden_gradient = hidden_error * self.activation_derivative(hidden_layer_output)

            # Atualiza os pesos
            self.weights_hidden_output += learning_rate * np.dot(hidden_layer_output.T, output_gradient)
            self.weights_input_hidden += learning_rate * np.dot(input_data.T, hidden_gradient)
            
    def predict(self, input_data):
        # Realiza a previsão
        hidden_layer_activation = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self.activation_function(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        output = self.activation_function(output_layer_activation)

        return output

    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        return x * (1 - x)