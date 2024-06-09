import streamlit as st
import numpy as np
import pandas as pd
st.write("Team: 22AIA-WEBSPIRITS")

st.title("Artificial Neural Network with Backpropagation")


# Define the sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)  # 4 neurons in the hidden layer
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backpropagation(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, epochs):
        for _ in range(epochs):
            self.feedforward()
            self.backpropagation()

# Generate the XOR dataset
data = {
    "input1": [0, 0, 1, 1],
    "input2": [0, 1, 0, 1],
    "output": [0, 1, 1, 0]
}
df = pd.DataFrame(data)

# Streamlit interface
st.write("Training Data (XOR problem):")
st.write(df)

# Prepare the data
X = df[["input1", "input2"]].values
y = df["output"].values.reshape(-1, 1)

# Create and train the neural network
nn = NeuralNetwork(X, y)
epochs = st.slider("Number of training epochs", min_value=1000, max_value=10000, step=1000, value=5000)
if st.button("Train Neural Network"):
    nn.train(epochs)
    st.write("Neural Network trained!")
    
    # Display the predictions
    nn.feedforward()
    st.write("Predicted Outputs after training:")
    st.write(nn.output)

# Classify a new sample
st.write("Classify a new sample")
input1 = st.number_input("Input 1", min_value=0.0, max_value=1.0, step=1.0, value=0.0)
input2 = st.number_input("Input 2", min_value=0.0, max_value=1.0, step=1.0, value=0.0)
if st.button("Classify"):
    new_sample = np.array([[input1, input2]])
    nn.input = new_sample
    nn.feedforward()
    st.write(f"Predicted class for ({input1}, {input2}): {nn.output[0][0]}")
