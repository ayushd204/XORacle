import numpy as np
import matplotlib.pyplot as plt 

class SimpleNeuralNetwork:
    def __init__(self,input_size, hidden_size, output_size,learning_rate=0.5,seed=42):
        #setting learning rate by 0.5!
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(1,hidden_size)
        self.bias_output = np.random.rand(1,output_size)

        self.errors = []

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    #activation function used! These are the ones that pass the values between the layers;
    
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
    #the x here is considered as the output of the sigmoid function but not like a literal x;

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_layer_input)

        return self.predicted_output
    
    def backward(self, X, Y):
        error = Y - self.predicted_output
        self.errors.append(np.mean(np.abs(error)))

        d_predicted_output = error * self.sigmoid_derivative(self.predicted_output)
        error_hidden_layer = d_predicted_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(self.hidden_layer_output)
        # Now updating the weights and biases ; this is the acutal learning part of the model;
        self.weights_hidden_output += self.hidden_layer_output.T.dot(d_predicted_output) * self.learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden_layer) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate
        self.bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, Y, epochs=10000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, Y)

    def predict(self, X):
        return self.forward(X)

    def plot_error(self):
        plt.plot(self.errors)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error During Training')
        plt.show()

    


