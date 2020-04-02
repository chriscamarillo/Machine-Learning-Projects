from random import random

# random range = r_max - r_min
# diff = amount values can differ from randomly generated line
# first_half + second_half = size of dataset
def generateDataset(r_min, r_max, diff, first_half, second_half, linearly_sep = True):
    if linearly_sep == True:
        # coeff[0] represents y = coeff[0]x + coeff[1]
        coeff = [random() * ((r_max - r_min + 1) - r_min) for x in range(2)]
        # [x, y, class]
        dataset = []
        i = 0
        while i < first_half:
            x = random() * diff
            y = x * coeff[0] + coeff[1] - random() * diff
            dataset.append([x, y, 0])
            i += 1
        
        j = 0
        while j < second_half:
            x = random() * diff
            y = x * coeff[0] + coeff[1] + random() * diff
            dataset.append([x, y, 1])
            j += 1

        return dataset

def generateDatasetFile(dataset, filename):
    f = open(f"datasets/{filename}", 'w')
    for v in dataset:
        f.write(','.join(str(i) for i in v) + '\n')


if __name__ == "__main__":
    d = generateDataset(1, 10, 25, 50, 50, True)
    for v in d:
        print(v)
    generateDatasetFile(d, "toyset.data")