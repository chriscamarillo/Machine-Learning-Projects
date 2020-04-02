from random import random

abs_min = -10**9
abs_max = 10**9

def normalize(d, x_min, x_max, y_min, y_max):
    norm = []
    for v in d:
        norm.append([(v[0]-x_min)/(x_max-x_min), (v[1]-y_min)/(y_max-y_min), v[2]])
    return norm


# random range = r_max - r_min
# diff = amount values can differ from randomly generated line
# first_half + second_half = size of dataset
def generateDataset(r_min, r_max, diff, first_half, second_half, linearly_sep = True):
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
            y = x * coeff[0] + coeff[1] - random() * diff
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
            y = x * coeff[0] + coeff[1] + random() * diff
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


if __name__ == "__main__":
    d = generateDataset(1, 10, 25, 50, 50, True)
    for v in d:
        print(v)
    #generateDatasetFile(d, "toyset.data")