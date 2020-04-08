from random import random, randint, sample, choices, shuffle
from math import *
import re

abs_min = -10**9
abs_max = 10**9
sign = [-1, 1]

# refactored generalized version of normalize
def normalize_dataset(dataset):
    # figure out maxes
    num_attr = len(dataset[0]) - 1
    attr_mins = []
    attr_maxs = []

    for i in range(num_attr):
        # figure out the maximum and minumum value for a certain attribute
        # AKA a column
        possible_vals = [d[i] for d in dataset]
        attr_mins.append(min(possible_vals))
        attr_maxs.append(max(possible_vals))
    
    for entry in dataset:
        for i in range(num_attr):
            entry[i] = 1 if attr_mins[i] == attr_maxs[i] else (entry[i]-attr_mins[i])/(attr_maxs[i]-attr_mins[i])

def normalize(d, x_min, x_max, y_min, y_max):
    norm = []
    for v in d:
        temp = []
        x = 1 if x_min == x_max else (v[0]-x_min)/(x_max-x_min)
        temp.append(x)
        y = 1 if y_min == y_max else (v[1]-y_min)/(y_max-y_min)
        temp.append(y)
        temp.append(v[2])
        norm.append(temp)
    return norm

def sprayer(m, b, lo, hi, border, noise=0):
    points_01 = [[], []]

    for i in range(lo + hi):
        x = random()
        y = x * m + b
        delta = 0
        if i < lo:
            delta = -random() * y - border
        else:
            delta = random() * (1 - y) + border

        y += delta
        c = 0 if i < lo else 1
        points_01[c].append([x, y, c])
 
    # for i in range((lo + hi) * 3):
    #     x, y = random(), random()
    #     c = 1 if y > x * m + b else 0
    #     points_01[c].append([x, y, c])

    dataset = []
    dataset += choices(points_01[0], k=lo)
    dataset += choices(points_01[1], k=hi)
    if noise > 0:
        shuffle(dataset)
        noise = [dataset.pop() for d in range(noise)]
        for n in noise:
            n[-1] = 1 if n[-1] == 0 else 0
        dataset += noise
        dataset.sort(key=lambda x: x[-1])

    return dataset

# random range = r_max - r_min
# diff = amount values can differ from randomly generated line
# first_half + second_half = size of dataset
def generateDataset(r_min, r_max, diff, bound, first_half, second_half, linearly_sep = True):
    if linearly_sep == True:
        # line[0] represents y = line[0]x + line[1]
        line = [sign[randint(0,len(sign)-1)] * random() * (r_max - r_min + 1) + r_min, random() * (r_max - r_min + 1) + r_min]
        # [x, y, class]
        dataset = []

        # [min, max]
        x_mm = [abs_max, abs_min]
        y_mm = [abs_max, abs_min]

        i = 0
        while i < first_half:
            x = random() * diff
            if x < x_mm[0]:
                x_mm[0] = x
            if x > x_mm[1]:
                x_mm[1] = x
            y = x * line[0] + line[1] - (random() * diff + bound)
            if y < y_mm[0]:
                y_mm[0] = y
            if y > y_mm[1]:
                y_mm[1] = y
            dataset.append([x, y, 0])
            i += 1

        j = 0
        while j < second_half:
            x = random() * diff
            if x < x_mm[0]:
                x_mm[0] = x
            if x > x_mm[1]:
                x_mm[1] = x
            y = x * line[0] + line[1] + (random() * diff + bound)
            if y < y_mm[0]:
                y_mm[0] = y
            if y > y_mm[1]:
                y_mm[1] = y
            dataset.append([x, y, 1])
            j += 1

        return normalize(dataset, x_mm[0], x_mm[1], y_mm[0], y_mm[1])


def generateDatasetFile(dataset, filename):
    f = open(f"datasets/{filename}", 'w')
    for v in dataset:
        f.write(','.join(str(i) for i in v) + '\n')


def undersample(dataset):
    class_0 = [v for v in dataset if v[-1] == 0] 
    class_1 = [v for v in dataset if v[-1] == 1]
    class_count_0 = len(class_0)
    class_count_1 = len(class_1)
    balanced = []

    if class_count_0 == class_count_1:
        return dataset
    elif class_count_0 < class_count_1: # truncate class_1 items to class_0 size
        balanced += class_0
        balanced += sample(class_1, k=class_count_0)
    else:
        balanced += sample(class_0, k=class_count_1)
        balanced += class_1

    a = [v for v in balanced if v[-1] == 0]
    b = [v for v in balanced if v[-1] == 1]
    print('undersample: class0: ', len(a))
    print('undersample: class1: ', len(b))

    return balanced


def oversample(dataset):
    class_0 = [v for v in dataset if v[-1] == 0] 
    class_1 = [v for v in dataset if v[-1] == 1]
    class_count_0 = len(class_0)
    class_count_1 = len(class_1)
    balanced = []
    
    if class_count_0 == class_count_1:
        return dataset
    elif class_count_0 < class_count_1:
        balanced += class_0 + choices(class_0, k=class_count_1 - class_count_0)
        balanced += class_1
    else:
        balanced += class_0
        balanced += class_1 + choices(class_1, k=class_count_0 - class_count_1)

    a = [v for v in balanced if v[-1] == 0]
    b = [v for v in balanced if v[-1] == 1]
    print('oversample: class0: ', len(a))
    print('oversample: class1: ', len(b))

    
    return balanced


if __name__ == "__main__":

    #c = generateDataset(-2, 2, 300, 10, 300, 700, True)
    #d = sprayer(random(), random() * 0.5, 200, 800)
    #e = sprayer(random(), random() * 0.5, 100, 900)
    #f = generateDataset(-2, 2, 300, 10, 10, 990, True)
    # generateDatasetFile(a, "50-50.data")
    # generateDatasetFile(b, "40-60.data")
    #generateDatasetFile(c, "30-70.data")
    #generateDatasetFile(d, "20-80.data")
    #generateDatasetFile(e, "10-90.data")
    #generateDatasetFile(f, "1-99.data")
    # nls = sprayer(1, 0, 300, 700, 0.05, noise=40)
    # generateDatasetFile(nls, "nls.data")
    pass