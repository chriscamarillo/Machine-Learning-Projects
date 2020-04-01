import re
import numpy as np
from os import listdir
from os.path import isfile, join

# assume last element is a binary class
def read_data(name):
    f = open(name, 'r')
    data = []

    for line in f.readlines():
        entry = re.split('\s|,', line)
        entry.pop(-1)
        data.append(entry)


    first_class = data[0][-1]

    # Mark differing class labels with 0's and 1's for the perceptron
    for i, entry in enumerate(data):
        entry[-1] = 0 if entry[-1] == first_class else 1
        data[i] = [float(attr) for attr in entry]

    return data

class Perceptron:
    def __init__(self, num_attr, learning_rate=0.05):
        # adds bias and initalizes weights between [-1, 1]
        self.weights = 2 * np.random.rand(num_attr + 1) - 1
        self.learning_rate = learning_rate

# Perceptron learning algorithm here
# returns if 1 if weights were updated and 0 if not
    def train(self, x, target):
        x_with_bias = np.append(1, np.array(x))
        hypothesis = self.evaluate(x)
        changes = self.learning_rate * (target - hypothesis) * x_with_bias
        self.weights = self.weights + changes
        return int(hypothesis != target)

    def evaluate(self, x):
        input_with_bias = np.append(1, np.array(x))
        return 1 if sum(self.weights * input_with_bias) > 0 else 0

def train_dataset(dataset):
    num_attr = len(dataset[0]) - 1
    p = Perceptron(num_attr)

    for epoch in range(1000):
        error_count = sum(p.train(entry[:-1], entry[-1]) for entry in dataset)
        print(error_count)

def main():
 # load in datasets
    path = 'datasets'
    files = [f for f in listdir(path) if isfile(join(path, f))]

    # for each dataset conduct the following tests
    #for f in files:
    dataset = read_data(join(path, files[0]))
    train_dataset(dataset)

if __name__ == '__main__':
   main()