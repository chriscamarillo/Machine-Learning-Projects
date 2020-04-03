from random import *
from math import *

abs_min = -10**9
abs_max = 10**9

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


# random range = r_max - r_min
# diff = amount values can differ from randomly generated line
# first_half + second_half = size of dataset
def generateDataset(r_min, r_max, diff, bound, first_half, second_half, linearly_sep = True):
    if linearly_sep == True:
        # coeff[0] represents y = coeff[0]x + coeff[1]
        coeff = [random() * (r_max - r_min + 1) + r_min for x in range(2)]
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
            y = x * coeff[0] + coeff[1] - (random() * diff + bound)
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
            y = x * coeff[0] + coeff[1] + (random() * diff + bound)
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
    dataset.sort(key = lambda x:x[-1])
    second_class_sum = sum(v[-1] for v in dataset)
    first_class_sum = len(dataset) - second_class_sum
    # more of class 1 than class 0
    if second_class_sum > first_class_sum:
        balanced = [v for v in dataset if v[-1] == 0]
        for i in [x+first_class_sum for x in range(first_class_sum)]:
            balanced.append(dataset[i])
    # more of class 0 than class 1 (or equal size)
    else:
        balanced = [v for v in dataset if v[-1] == 1]
        for i in range(second_class_sum):
            balanced.insert(0, dataset[i])
    return balanced


def oversample(dataset):
    dataset.sort(key = lambda x:x[-1])
    second_class_sum = sum(v[-1] for v in dataset)
    first_class_sum = len(dataset) - second_class_sum
    # more of class 1 than class 0
    if second_class_sum > first_class_sum:
        r = [dataset[randint(0, first_class_sum-1)] for i in range(second_class_sum-first_class_sum)]
        balanced = r + dataset
    # more of class 0 than class 1 (or equal size)
    else:
        r = [dataset[randint(second_class_sum-1,len(dataset)-1)] for i in range(first_class_sum-second_class_sum)]
        balanced = dataset + r
    return balanced


if __name__ == "__main__":
    a = generateDataset(1, 10, 25, 5,  500, 500, True)
    b = generateDataset(1, 10, 25, 5, 400, 600, True)
    c = generateDataset(1, 10, 25, 5, 300, 700, True)
    d = generateDataset(1, 10, 25, 5, 200, 800, True)
    e = generateDataset(1, 10, 25, 5, 100, 900, True)
    f = generateDataset(1, 10, 25, 5, 10, 990, True)
    generateDatasetFile(a, "50-50.data")
    generateDatasetFile(b, "40-60.data")
    generateDatasetFile(c, "30-70.data")
    generateDatasetFile(d, "20-80.data")
    generateDatasetFile(e, "10-90.data")
    generateDatasetFile(f, "1-99.data")