from os import listdir
from os.path import isfile, join
from perceptron import *

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



if __name__ == '__main__':
    # load in datasets 
    path = 'datasets'
    files = [f for f in listdir(path) if isfile(join(path, f))]

    # for f in files:
    dataset = read_data(join(path, files[1]))
    num_attr = len(dataset[0]) - 1 # class domain takes up one column
    p = Perceptron(num_attr)
    epochs = p.train_dataset(dataset)
    print(F'{files[1]} collapsed at {epochs} epochs')
