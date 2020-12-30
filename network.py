import numpy as np

class Network:

    def __init__(self, layers_sizes, alpha = 0.01):
        # layers_sizes is an array how how big each layer is
        self.layers_size = layers_sizes

        # set alpha or step amount
        self.alpha = alpha

        # create new randomized layers (depending on sizes)
        self.layers = []
        for i in range(len(self.layers_size)):
            self.layers.append(np.random.random(size=self.layers_size[i]))

        # initialize random weights
        self.weights = []
        for i in range(1, len(self.layers_size)):
            self.weights.append(np.random.random(size=(self.layers_size[i], self.layers_size[i-1])))

        # initialize biases of 0
        self.biases = []
        for i in range(len(self.layers_size)):
            self.biases.append(np.zeros(self.layers_size[i]))

    # use a certain input to train
    # input is a vector for the input and a vector for the expected output
    # returns the error for the prediction (sum of squares)
    def backprop_on_input(self, vec_in, goal_out):

        if (len(vec_in) != self.layers[0].size):
            raise TypeError
        if (len(goal_out) != self.layers[-1].size):
            raise TypeError

        # propagate forward
        self.run_on_input(vec_in)

        # inital delta
        end_delta = self.layers[-1] - goal_out
        end_error = end_delta**2 

        # initialize lists of layer deltas and weight deltas
        deltas = []
        for i in range(len(self.layers_size)):
            deltas.append(np.zeros(self.layers_size[i]))
        deltas[-1] = end_delta
        
        weight_deltas = []
        for i in range(len(self.weights)):
            weight_deltas.append(np.zeros(self.weights[i].shape))

        # main backpropagation loop
        for i in range(len(self.weights), 0, -1):
            # calculate previous layer's weight delta (for the weights behind the layer)
            weight_deltas[i-1] = np.outer(deltas[i], self.layers[i-1])
            # adjust the weights
            self.weights[i-1] = self.weights[i-1] - (self.alpha * weight_deltas[i-1])
            # get the layer delta before that
            deltas[i-1] = np.dot(deltas[i], self.weights[i-1])
        
        # return error
        return sum(end_error)
            

        

        

    # use the current model to produce output
    # input is a vector (as a regular python list)
    def run_on_input(self, vec_in):
        # make sure input is of correct length
        if (len(vec_in) != self.layers[0].size):
            raise TypeError

        # put input in first layer
        self.layers[0] = np.array(vec_in)

        # propagate forward
        relu_func = np.vectorize(relu)
        for i in range(1, len(self.layers_size)):
            current = relu_func(np.dot(self.weights[i-1], self.layers[i-1].T) + self.biases[i])
            self.layers[i] = current

        # return the output
        output = np.copy(self.layers[-1])
        return output

# activation function
def relu(x):
    if x < 0:
        return 0
    else:
        return x
