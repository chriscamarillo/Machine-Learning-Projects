import re
import numpy as np
from random import random

class Perceptron:
    # adds bias, initalize weights between [-1, 1], and set the learning rate
    def __init__(self, num_attr, learning_rate=0.01, max_epochs=1000):
        self.weights = 2 * np.random.rand(num_attr + 1) - 1
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
    def evaluate(self, x):
        input_with_bias = np.append(1, np.array(x))
        return 1 if sum(self.weights * input_with_bias) > 0 else 0

    # Perceptron learning algorithm here
    # returns True if perceptron guessed incorrectly (changes made)
    def update_weights(self, x, target):
        x_with_bias = np.append(1, np.array(x))
        hypothesis = self.evaluate(x)
        changes = self.learning_rate * (target - hypothesis) * x_with_bias
        self.weights = self.weights + changes
        return int(hypothesis != target)

    def train_dataset(self, dataset):
        # keep training til the perceptron correctly classifies all the examples
        epochs = 0
        needs_training = True
        
        errors_on_epoch = []
        while needs_training and epochs < self.max_epochs:
            num_errors = 0
            needs_training = False
            for entry in dataset:
                x_input = entry[:-1]
                target = entry[-1]
                messed_up = self.update_weights(x_input, target)
                if messed_up:
                    needs_training = True
                    num_errors += 1
            epochs += 1
            errors_on_epoch.append(num_errors)

        return epochs, errors_on_epoch

    # returns error per class
    # THIS is the one doing the plots 
    def test_dataset(self, dataset):
        # each element corresponding to a FALSE {1: POS, 0 NEG}
        error_per_class = [0, 0]
        for entry in dataset:
            x_input = entry[:-1]
            target = int(entry[-1])
            hypothesis = self.evaluate(x_input)
            if hypothesis != target:
                error_per_class[hypothesis] += 1 
        
        return error_per_class