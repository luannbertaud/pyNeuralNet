#!/bin/env python3

import time
import datetime
import random
import numpy as np
import copy as cp
from sys import stdout
from matplotlib import pyplot as plt, image as plt_img

import beepy
from pprint import pprint


error = []
class NeuralNetwork():

    def __init__(self, layers_sizes):
        weight_shapes = [(a,b) for a,b in zip(layers_sizes[1:],layers_sizes[:-1])]
        self.weights = [np.random.standard_normal(s)/s[1]**0.5 for s in weight_shapes]
        self.biases = [np.zeros((s,1)) for s in layers_sizes[1:]]

    def sigmoid(self, x, zero_centered=True):
        if zero_centered:
            return (1/(1+np.exp(-x)))-0.5
        return 1/(1+np.exp(-x))
    
    def rev_sigmoid(self, x, zero_centered=True):
        if zero_centered:
            return np.log((x+0.5)/(1-(x+0.5)))
        return np.log(x/(1-x))

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
    
    def old_backward_prop(self, x, calc_proba_from):
        # vals = self.rev_sigmoid(x)
        self.percentages = [np.zeros(i.shape) for i in self.weights]

        self.neuron_adapted = cp.deepcopy(self.neuron_values.tolist())
        self.neuron_adapted.insert(0, calc_proba_from)
        self.neuron_adapted = np.array(self.neuron_adapted)
        

        for i in range(len(self.weights)):
            i = len(self.weights) - i - 1
            for y in range(len(self.weights[i])):
                self.neuron_adapted[i+1][y] = self.rev_sigmoid(self.neuron_adapted[i+1][y])
                for z in range(len(self.weights[i][y])):
                    part = self.neuron_adapted[i][z] * self.weights[i][y][z]
                    self.percentages[i][y][z] = part/self.neuron_adapted[i+1][y]
        
        # print('-'*50)
        self.neuron_adapted = [np.zeros(i.shape) for i in self.neuron_adapted]
        self.neuron_adapted[len(self.neuron_adapted)-1] = x
        # print(self.neuron_adapted)
        # print('--')
        # print(self.neuron_values)
        # print('--')
        # print(self.percentages)           
        for i in range(len(self.weights)):
            i = len(self.weights) - i - 1
            for y in range(len(self.weights[i])):
                for z in range(len(self.weights[i][y])):
                    self.neuron_adapted[i][z] += self.percentages[i][y][z] * self.rev_sigmoid(self.neuron_adapted[i+1][y]) / self.weights[i][y][z]
            for y in range(len(self.neuron_adapted[i])):
                self.neuron_adapted[i][y] = self.neuron_adapted[i][y] / len(self.weights[i])

        return self.neuron_adapted

    def backward_prop(self, inputs, labels):
        global error
        # lr = 0.1

        results = self.forward_prop(inputs)
        # results = np.round(results, 2) 
        
        pprint(results)
        print('¨'*5)

        # self.weights_c = cp.deepcopy(self.weights)
        # self.biases_c = cp.deepcopy(self.biases)

        self.neuron_adapted = cp.deepcopy(self.neuron_values.tolist())
        self.neuron_adapted.insert(0, inputs.tolist())
        self.neuron_adapted = np.array(self.neuron_adapted)
        self.neuron_adapted = [np.asarray(i) for i in self.neuron_adapted]

        self.errors = [np.zeros(np.array(i).shape) for i in self.neuron_adapted]
        error = np.subtract(labels, results)
        self.errors[len(self.errors)-1] = error

        for i in range(len(self.neuron_adapted)):
            i = len(self.neuron_adapted) - i - 1

            if i <= 0:
                break
        
            # gradient = self.neuron_values[i] * (1 - self.neuron_values[i])
            # gradient = np.multiply(gradient, error)
            # gradient = np.multiply(gradient, lr)

            # neuron_t = np.transpose(self.neuron_values[i-1])
            # weight_deltas = np.multiply(neuron_t, gradient)
            # self.weights_c[i] = np.add(self.weights_c[i], weight_deltas)
            # self.biases_c[i] = np.add(self.biases_c[i], gradient)
            
            # pprint(error)
            # print(self.neuron_adapted[i])
            # print('A')
            for y in range(len(self.weights[i-1])):
                for w in range(len(self.weights[i-1][y])):
                    print(i, '   ', y)
                    print(self.errors[i][y], '     ', self.neuron_adapted[i][y])
                    
                    self.errors[i-1][w] += self.rev_sigmoid(self.errors[i][y], True) * (self.neuron_adapted[i-1][w] / self.rev_sigmoid(self.neuron_adapted[i][y], True))


            # weights_t = np.transpose(self.weights[i])
            # tmp_error = []
            # for i in range(len(weights_t)):
            #     error_sum = 0
            #     for y in range(len(weights_t[i])):
            #         error_sum += weights_t[i][y] * error[y]
            #         print(f'--#---{error_sum}---#--')
            #     tmp_error.append(error_sum)
            #     print('~~~~~')
            # error = np.array(tmp_error)


        # pprint(error)
        return np.add(self.errors[0], inputs)

        # for e in error:
        #     print(self.sigmoid(e[0]))


        # hidden_d = self.neuron_values[0] * (1 - self.neuron_values[0])
        # hidden_e = np.multiply(hidden_d, error)
        # hidden_gradient = np.multiply(hidden_e, lr)

        # input_t = np.transpose(inputs)
        # weight_ih_deltas = np.multiply(input_t, hidden_gradient)
        # self.weights_c[0] = np.add(self.weights_c[0], weight_ih_deltas)
        # self.biases_c[0] = np.add(self.biases_c[0], hidden_gradient)
    
    def print_accuracy(self, inputs, answers):
        correct = 0 
        for i in range(len(inputs)):
            result = self.forward_prop(inputs[i])
            if (np.argmax(result) == np.argmax(answers[i])):
                correct += 1
        print("Accuracy: {0:}/{1:}\t{2:.2f}%".format(correct, len(inputs), ((correct/len(inputs))*100)))

    def one_train(self, inputs, labels):
        global error
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

    def training(self, inputs, answers, iteration, sound=False):
        iter_delta = 0
        iter_nb = 0
        iter_duration = 0.1
        iter_start = time.time()
        
        print('#'*40)
        self.print_accuracy(inputs, answers)
        for y in range(iteration):

            stdout.write("\rIteration {0:}/{1:}\t{2:.2f}%\t\tETA:{3:}".format(y+1, iteration, ((y+1)/iteration)*100, datetime.timedelta(seconds=int(iter_duration*(iteration-y+1)))))
            stdout.flush()

            i = random.randint(0,len(inputs)-1)
            self.one_train(inputs[i], answers[i])
            
            iter_delta = time.time() - iter_start
            iter_nb += 1
            if iter_delta >= 2:
                iter_duration = iter_delta / iter_nb
                iter_start = time.time()
                iter_nb = 0


        print("\n")
        self.print_accuracy(inputs, answers)
        print('#'*40)
        if sound:
            beepy.beep(sound=1)
            beepy.beep(sound=1)

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
    
    plt_show_img(aimg)
    return np.argmax(NN.forward_prop(aimg))






with np.load('data/mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

"""
data = [np.array([[0],[1]]),
       np.array([[1],[0]]),
       np.array([[1],[1]]),
       np.array([[0],[0]])]

answer = [np.array([[1]]),
         np.array([[1]]),
         np.array([[0]]),
         np.array([[0]])]
"""


data = training_images
answer = training_labels

np.random.seed(8)

# layers_sizes = (784, 256, 128, 64, 10)
layers_sizes = (2, 2, 1)
NN = NeuralNetwork(layers_sizes)

# NN.load_network()

# for i in range(len(data)):
#     print(NN.forward_prop(data[i]))

# NN.training(training_images, training_labels, 5000, sound=True)
#NN.one_train(training_images[0], training_labels[0])

# for i in range(len(data)):
#     print(NN.forward_prop(data[i]))

# print(try_on_image('image0.png'))

# NN.export_network()

# print(np.argmax(NN.forward_prop(data[0])))

# entry = np.array([[8], [2], [3]])
entry = np.array([[0.15], [0.007]])

# NN.weights = [np.array([[1.22, 1.55]])]

passed = NN.forward_prop(entry)


print('-----Neurons-----')
pprint(NN.neuron_values)
print('-----Biases-----')
pprint(NN.biases)
print('-----Weight-----')
pprint(NN.weights)

print('-----BACK-----')

data = NN.backward_prop(entry, np.array([[0.40]]))
print('='*40)
print('-----res-----')
entry = np.add(entry, data)
print(entry)
NN.forward_prop(entry)
print('-----Neurons-----')
pprint(NN.neuron_values)
# print('-----Biases-----')
# pprint(NN.biases)
# print('-----Weight-----')
# pprint(NN.weights)