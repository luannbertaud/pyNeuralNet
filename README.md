[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fluannbertaud%2FpyNeuralNet&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# What is pyNeuralNet ?

This project contain what you need to quickly set up, train and use a neural network. The model is trained on a supervised way.
Once trained the network can be exported then reloaded during an other session, this allow you use the network on computer without any training.

# Use
The `NN2_0.py` module contain a class named `NeuralNetwork`, that's all you need to import.

## Creation

    layers_shape = (2, 3, 3, 1)
    NN = NeuralNetwork(layers_shapes)
This will create a neural network made of 2 input neurons, 2 hidden layers of 3 neurons each, and 1 output neuron.

## Training

    data = [np.array([[0],[1]]),
            np.array([[1],[0]]),
            np.array([[1],[1]]),
            np.array([[0],[0]])]
    
    answers = [np.array([[1]]),
               np.array([[1]]),
               np.array([[0]]),
               np.array([[0]])]
    
    NN.training(data, answers, 100000)

Here we want to train the network on a logical 'OR'. As the training is supervised, you must provide the correct answers to the model.
Then you supply the number of training cycle you want, a cycle stand for a try->correction of the model.

## Prediction

    NN.forward_prop(data[0])

To make a prediction, use the forward propagation function with the input neurons value as parameter. This function will return a array containing values of the outputs neurons.

In case of multiple output neurons (ex: digit prediction) you can select the most activated neuron by passing the result of the `forward_prop` function in `numpy.argmax()`.

## Import / Export

    NN.load_network()
    NN.export_network()

These functions will create or ask for `weights.npy` and `biases.npy` files in the current folder, they represent the model shape and values.
