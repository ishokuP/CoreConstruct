import pickle

def load_weights_biases(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data['W1'], data['b1'], data['W2'], data['b2']

def relu(x):
    return max(0, x)

def forward_pass_hidden(W1, b1, inputs):
    hidden_outputs = []
    for j in range(len(W1[0])):
        net_input = sum(inputs[i] * W1[i][j] for i in range(len(inputs))) + b1[j]
        hidden_outputs.append(relu(net_input))
    return hidden_outputs

def forward_pass_output(W2, b2, hidden_outputs):
    output = []
    for j in range(2):  # Assuming the model has 2 outputs
        net_input = sum(hidden_outputs[i] * W2[i + j * len(hidden_outputs)] for i in range(len(hidden_outputs))) + b2[j]
        output.append(relu(net_input))
    return output

def test_model(input_data):
    # Load the weights and biases
    W1, b1, W2, b2 = load_weights_biases('models/ann/annweights.pkl')
    
    # Perform a forward pass with the input data
    hidden_outputs = forward_pass_hidden(W1, b1, input_data)
    output = forward_pass_output(W2, b2, hidden_outputs)
    
    return output

if __name__ == "__main__":
    # Example input data (adjust based on your model's expected input format)
    single_input = [1, 2, 3]  # Replace with the actual input format expected by your model
    print("TRUE")
    
    result = test_model(single_input)
    print(f"Test input: {single_input}, Model output: {result}")