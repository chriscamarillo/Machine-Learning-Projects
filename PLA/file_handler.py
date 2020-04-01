import re
from os import listdir
from os.path import isfile, join

def read_data(name):
    f = open(name, 'r')
    data = []

    for line in f.readlines():
        entry = re.split('\W', line)
        data.append(entry)

    return data

if __name__ == '__main__':
    # load in datasets
    path = 'datasets'
    files = [f for f in listdir(path) if isfile(join(path, f))]