#!/bin/env python3

import time
import beepy
import random
import numpy as np
import copy as cp
from sys import stdout
from matplotlib import pyplot as plt, image as plt_img


def convert_sec(seconds):
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def plt_show_img(aimg, timeout=0):
    imgplot = plt.imshow(np.reshape(np.asarray(aimg), (28, 28)), cmap='Greys')
    if timeout != 0:
        plt.show(block=False)
        plt.pause(timeout)
        plt.close()
    else:
        plt.show()


def try_on_image(file):
    img = plt_img.imread(file)
    aimg = np.asarray(img)
    aimg = aimg.reshape(784, 1)

    return np.argmax(NN.forward_prop(aimg))



class NeuralNetwork():

    def __init__(self, layers_sizes):
        np.seterr(over='ignore')
        weight_shapes = [(a,b) for a,b in zip(layers_sizes[1:],layers_sizes[:-1])]
        self.weights = [np.random.standard_normal(s)/s[1]**0.5 for s in weight_shapes]
        self.biases = [np.zeros((s,1)) for s in layers_sizes[1:]]


    def sigmoid(self, x, zero_centered=False):
        if zero_centered:
            return (1/(1+np.exp(-x)))-0.5
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


    def print_accuracy(self, inputs, answers):
        correct = 0 
        for i in range(len(inputs)):
            result = self.forward_prop(inputs[i])
            if (np.argmax(result) == np.argmax(answers[i])):
                correct += 1
        print("Accuracy: {0:}/{1:}\t{2:.2f}%".format(correct, len(inputs), ((correct/len(inputs))*100)))


    def forward_prop(self, x):
        self.neuron_values = []
        for w,b in zip(self.weights, self.biases):
            x = self.sigmoid(np.matmul(w, x) + b)
            self.neuron_values.append(x)
        self.neuron_values = np.array(self.neuron_values, dtype=object)
        return x


    def one_train(self, inputs, labels, learning_rate):
        results = self.forward_prop(inputs)
        error = np.subtract(labels, results)

        for i in range(len(self.neuron_values)):
            i = len(self.neuron_values) - i - 1
            if (i <= 0):
                break
        
            gradient = self.neuron_values[i] * (1 - self.neuron_values[i])
            gradient = np.multiply(gradient, error)
            gradient = np.multiply(gradient, learning_rate)

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
        hidden_gradient = np.multiply(hidden_e, learning_rate)

        input_t = np.transpose(inputs)
        weight_ih_deltas = np.multiply(input_t, hidden_gradient)
        self.weights[0] = np.add(self.weights[0], weight_ih_deltas)
        self.biases[0] = np.add(self.biases[0], hidden_gradient)


    def train(self, inputs, answers, iteration, learning_rate=0.1, sound=False):
        iter_delta = 0
        iter_nb = 0
        iter_duration = 0.1
        iter_start = time.time()
        
        print('#'*40)
        self.print_accuracy(inputs, answers)
        for y in range(iteration):

            stdout.write("\rIteration {0:}/{1:}\t{2:.2f}%\t\tETA:{3:}"\
            .format(y+1, iteration, ((y+1)/iteration)*100, convert_sec(int(iter_duration*(iteration-y+1)))))
            stdout.flush()

            i = random.randint(0,len(inputs)-1)
            self.one_train(inputs[i], answers[i], learning_rate)
            
            iter_delta = time.time() - iter_start
            iter_nb += 1
            if iter_delta >= 2:
                iter_duration = iter_delta / iter_nb
                iter_start = time.time()
                iter_nb = 0

        print("")
        self.print_accuracy(inputs, answers)
        print('#'*40)
        if sound:
            beepy.beep(sound=1)
            beepy.beep(sound=1)



if __name__ == "__main__":

    with np.load('data/mnist.npz') as data:
        training_images = data['training_images']
        training_labels = data['training_labels']

    # np.random.seed(8)

    layers_sizes = (784, 20, 10, 10)
    NN = NeuralNetwork(layers_sizes)
    # NN.load_network()

    NN.train(training_images, training_labels, 5000, learning_rate=0.1, sound=False)

    # plt_show_img(training_images[0])
    # print(try_on_image('image0.png'))

    # NN.export_network()