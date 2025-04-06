import numpy as np
from NN import SimpleNeuralNetwork


X = np.array([[0,0], [1,0], [0,1], [1,1]])
Y = np.array([[0], [1], [1], [0]])  
#XOR problem not being linearly separable and so used as an example;

nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1)
nn.train(X, Y, epochs=10000)
nn.plot_error()

print("Predictions:")
print(nn.predict(X))

#simplest form of neural network implementation;