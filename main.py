from network import Network
import numpy as np

if __name__ == "__main__":
    new_network = Network([2,5,2], 0.05)
    # print(new_network.layers_size)
    # print(new_network.layers)
    # print(new_network.weights)
    # print(new_network.biases)
    
    inp = [[1, 0], [0, 1], [1, 1]]
    goal = [[0, 1], [0, 0], [1, 0]]
    

    for i in range(200):
        for j in range(len(inp)):
            this_run_error = new_network.backprop_on_input(inp[j], goal[j])
            

    for i in range(len(inp)):
        new_network.run_on_input(inp[i])
        out = new_network.layers[-1]
        print('\nexpected:{}\nobtained:{}'.format(goal[i], out))