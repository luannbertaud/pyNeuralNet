import numpy as np
import matplotlib.pyplot as plt
import random
import copy as cp
from sys import stdout


class NeuralNetwork():

    def __init__(self, layers_sizes):
        weight_shapes = [(a,b) for a,b in zip(layers_sizes[1:],layers_sizes[:-1])]
        self.weights = [np.random.standard_normal(s)/s[1]**0.5 for s in weight_shapes]
        self.biases = [np.zeros((s,1)) for s in layers_sizes[1:]]

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def load_network(self):
        w = []
        with open("weigths.npy", 'rb') as f:
            while 1:
                try:
                    w_t = np.load(f)
                    w.append(w_t)
                except:
                    break
        self.weights = cp.deepcopy(w)

        b = []
        with open("biases.npy", 'rb') as f:
            while 1:
                try:
                    b_t = np.load(f)
                    b.append(b_t)
                except:
                    break
        self.biases = cp.deepcopy(b)

    def export_network(self):
        with open("biases.npy", 'wb') as f:
            for b in self.biases:
                np.save(f, b)

        with open("weigths.npy", 'wb') as f:
            for w in self.weights:
                np.save(f, w)

    def forward_prop(self, x):
        self.neuron_values = []
        for w,b in zip(self.weights, self.biases):
            x = self.sigmoid(np.matmul(w, x) + b)
            self.neuron_values.append(x)
        self.neuron_values = np.array(self.neuron_values)
        return x
    
    def print_accuracy(self, inputs, answers):
        correct = 0 
        for i in range(len(inputs)):
            result = self.forward_prop(inputs[i])
            if (np.argmax(result) == np.argmax(answers[i])):
            #if (answers[i][0] - result < 0.1):
                correct += 1
        print("Accuracy: {0:}/{1:}\t{2:}%".format(correct, len(inputs), ((correct/len(inputs))*100)))

    def one_train(self, inputs, labels):
        lr = 0.1

        results = self.forward_prop(inputs)
        error = np.subtract(labels, results)

        for i in range(len(self.neuron_values)):
            i = len(self.neuron_values) - i - 1
            if (i <= 0):
                break
        
            gradient = self.neuron_values[i] * (1 - self.neuron_values[i])
            gradient = np.multiply(gradient, error)
            gradient = np.multiply(gradient, lr)

            neuron_t = np.transpose(self.neuron_values[i-1])
            weight_deltas = np.multiply(neuron_t, gradient)
            self.weights[i] = np.add(self.weights[i], weight_deltas)
            self.biases[i] = np.add(self.biases[i], gradient)

            weights_t = np.transpose(self.weights[i])
            tmp_error = []
            for i in range(len(weights_t)):
                error_sum = 0
                for y in range(len(weights_t[i])):
                    error_sum += weights_t[i][y] * error[y]
                tmp_error.append(error_sum)
            error = np.array(tmp_error)


        hidden_d = self.neuron_values[0] * (1 - self.neuron_values[0])
        hidden_e = np.multiply(hidden_d, error)
        hidden_gradient = np.multiply(hidden_e, lr)

        input_t = np.transpose(inputs)
        weight_ih_deltas = np.multiply(input_t, hidden_gradient)
        self.weights[0] = np.add(self.weights[0], weight_ih_deltas)
        self.biases[0] = np.add(self.biases[0], hidden_gradient)

    def training(self, inputs, answers, iteration):
        self.print_accuracy(inputs, answers)
        for y in range(iteration):
            stdout.write("\rIteration {0:}/{1:}\t{2:.2f}%".format(y, iteration, (y/iteration)*100))
            stdout.flush()
        
            i = random.randint(0,len(training_images)-1)
            #NN.train(data[i], answer[i])
            self.one_train(inputs[i], answers[i])
        print("\n")
        self.print_accuracy(inputs, answers)








with np.load('data/mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']


data = [np.array([[0],[1]]),
        np.array([[1],[0]]),
        np.array([[1],[1]]),
        np.array([[0],[0]])]

answer = [np.array([[1]]),
          np.array([[1]]),
          np.array([[0]]),
          np.array([[0]])]


#plt.imshow(training_images[0].reshape(28,28), cmap="gray")
#plt.show()

layers_sizes = (784, 10, 10, 10,10)
NN = NeuralNetwork(layers_sizes)

#print(NN.forward_prop(data[0]),"\n\n")
#print(NN.forward_prop(data[1]),"\n\n")
#print(NN.forward_prop(data[2]),"\n\n")
#print(NN.forward_prop(data[3]),"\n\n")
