from os import listdir
from os.path import isfile, join
from perceptron import *
from random import shuffle, seed
from linearly_sep import generateDataset
import matplotlib.pyplot as plt

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

def stratify(dataset):
    train_len = 0.2

    # sort T by classifier (last entry)
    dataset.sort(key = lambda t:t[-1])

    # split T into x lists (x = amount of classes) -> call class_split
    class_split = [[x for x in dataset if x[-1] == c] for c in [0, 1]]
    
    # shuffle each class list
    for c in class_split:
        seed()
        shuffle(c)

    # pull 20% of dataset and use as training set (of each classifier)
    test = []

    for c in class_split:
        for i in range(int(len(c) * train_len)):
            test.append(c.pop())
            
    # other 80% is testing set
    train = []
    for i in class_split:
        for j in i:
            train.append(j)

    return (train, test)

if __name__ == '__main__':
    # load in datasets 
    path = 'datasets'
    files = [f for f in listdir(path) if isfile(join(path, f))]

    # toy domain test first x: [-1, 1] difference of 0.5
    # toy_domain = generateDataset(-1, 1, 0.5, 1000, 1000)



    for f in files:
        dataset = read_data(join(path, f))
        num_attr = len(dataset[0]) - 1 # class domain takes up one column
        p = Perceptron(num_attr)
        epochs = p.train_dataset(dataset)
        print(F'{files[1]} collapsed at {epochs} epochs')