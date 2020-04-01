from random import random

def generateDataset(start, end, diff, lo, hi, linearly_sep = True):
    if linearly_sep == True:
        # coeff[0] represents y = coeff[0]x + coeff[1]
        coeff = [random() * ((end - start + 1) - start) for x in range(2)]
        # [x, y, class]
        dataset = []
        i = 0
        while i < lo:
            x = random() * diff
            y = x * coeff[0] + coeff[1] + random() * diff
            dataset.append([x, y, 0])
            i += 1
        
        j = 0
        while j < hi:
            x = random() * diff
            y = x * coeff[0] + coeff[1] + random() * diff
            dataset.append([x, y, 1])
            j += 1

        return dataset


def main():
    pipi = generateDataset(1, 10, 25, 50, 50, True)
    for v in pipi:
        print(v)

if __name__ == "__main__":
    main()