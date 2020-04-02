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

# returns EPOCHS and the number of individual training and 
# testing errors per class
def run_simulation(train, test):
    num_attr = len(train[0]) - 1 # exclude class marker
    p = Perceptron(num_attr)
    epochs = p.train_dataset(train)
    
    # gets the number of errors per class
    training_errors = p.test_dataset(train) 
    testing_errors = p.test_dataset(test)

    print(epochs, training_errors, testing_errors)
    return (epochs, training_errors, testing_errors)



if __name__ == '__main__':
    # load in datasets 
    path = 'datasets'
    files = [f for f in listdir(path) if isfile(join(path, f))]

    for f in files:
        # set up data and run_simulation
        dataset = read_data(join(path, f))
        train, test = stratify(dataset)
        epochs, training_errors, testing_errors = run_simulation(train, test)
        
        print(f)

        # training report
        # TODO: plot this for pics
        train_error_rate = sum(training_errors) / len(train) * 100
        train_c1_count = sum(t[-1] for t in train)
        train_c0_count = len(train) - train_c1_count
        train_c1_error_rate = training_errors[1] / train_c1_count * 100
        train_c0_error_rate = training_errors[0] / train_c0_count * 100

        print(F'Training report: took {epochs} epochs to land at {train_error_rate}% total error rate')
        print(F'Class 0 had {training_errors[0]} out of {train_c0_count} errors: {train_c0_error_rate}%')
        print(F'Class 1 had {training_errors[1]} out of {train_c1_count} errors: {train_c1_error_rate}%')
        

        # testing report
        # TODO: plot this for pics
        test_error_rate = sum(testing_errors) / len(test) * 100
        test_c1_count = sum(t[-1] for t in test)
        test_c0_count = len(test) - test_c1_count
        test_c1_error_rate = testing_errors[1] / test_c1_count * 100
        test_c0_error_rate = testing_errors[0] / test_c0_count * 100

        print(F'Testing report: {test_error_rate}% total error rate')
        print(F'Class 0 had {testing_errors[0]} out of {test_c0_count} errors: {test_c0_error_rate}%')
        print(F'Class 1 had {testing_errors[1]} out of {test_c1_count} errors: {test_c1_error_rate}%')
        