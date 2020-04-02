import re
import numpy as np

class Perceptron:
    # adds bias, initalize weights between [-1, 1], and set the learning rate
    def __init__(self, num_attr, learning_rate=0.01):
        self.weights = 2 * np.random.rand(num_attr + 1) - 1
        self.learning_rate = learning_rate

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

    def train_dataset(self, dataset, max_epochs=1000):
        # keep training til the perceptron correctly classifies all the examples
        epochs = 0
        needs_training = True
        while needs_training and epochs < max_epochs:
            needs_training = False
            for entry in dataset:
                x = entry[:-1]
                target = entry[-1]
                messed_up = self.update_weights(x, target)
                if messed_up:
                    needs_training = True
            
            epochs += 1
        return epochs

    # returns error per class
    def test_dataset(self, dataset):
        error_per_class = [0, 0]
        for entry in dataset:
            x = entry[:-1]
            target = int(entry[-1])
            hypothesis = self.evaluate(x)
            if hypothesis != target:
                error_per_class[target] += 1 
        
        return error_per_class