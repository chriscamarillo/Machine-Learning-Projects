import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('QT4Agg')

from random import random

# thank you stack overflow for this solid
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, '--')

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
        
        # plot if we got 2D coords + class
        if len(dataset[0]) == 3:
            error_on_epoch = []

            while needs_training and epochs < max_epochs:
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
                error_on_epoch.append(num_errors)



            # picture
            fig, ax = plt.subplots(2)

            for entry in dataset:
                ax[0].plot(entry[0], entry[1], 'o', color='orange' if entry[2] == 0 else 'green')

            # ax.plot([0, 1], [(self.weights[0] + self.weights[1]) / self.weights[2], 
            #     self.weights[0] / self.weights[2]])
            slope = -self.weights[1] / self.weights[2]
            intercept = -self.weights[0] / self.weights[2]
            ax[0].set_title(F'Perceptron\'s guess on how to split after {epochs} epochs')
            ax[1].set_title(F'error rate improvement over time')


            # plot perceptrons guess
            x_vals = np.array(ax[0].get_xlim())
            y_vals = intercept + slope * x_vals
            ax[0].plot(x_vals, y_vals)
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('y')
            ax[0].grid(True)

            # plot Improvement (epoch vs error rate)
            ax[1].plot(np.arange(len(error_on_epoch)) + 1, [e_c / len(dataset) * 100 for e_c in error_on_epoch])
            ax[1].set_xlabel('epoch')
            ax[1].set_ylabel('error rate')
            min_epoch_y = min(error_on_epoch)
            min_epoch_x = error_on_epoch.index(min_epoch_y) 
            ax[1].annotate(F'({min_epoch_x}, {min_epoch_y / len(dataset) * 100})', xy=(min_epoch_x, min_epoch_y), textcoords='data')
            ax[1].grid(True)
            plt.show()
        else:
            print('no animation available! dataset is comprised of more than 2 attributes.')
            print('going the console way...')

            while needs_training and epochs < max_epochs:
                needs_training = False
                for entry in dataset:
                    x_input = entry[:-1]
                    target = entry[-1]
                    messed_up = self.update_weights(x_input, target)
                    if messed_up:
                        needs_training = True
                epochs += 1

        print(epochs)
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