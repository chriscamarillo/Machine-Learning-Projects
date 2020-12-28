'''
    Authors: chriscamarillo, aig77, juanSLopez
    ECE548 Project 1
    Description:
        K-NN classifiers, weighted distances. Using three domains from the UCI
        repository,compare the testing-set performance of two k-NN approaches: one
        without example weighting, and the other with example weighting.
    
    2/2/2020
'''

import math
from random import shuffle, seed
 
''' 
    * T = training set
    * x = unknown instance
    * k = number of nearest neighbors
    * The last element of an example is assumed to be its class
      therefore x's last element should be ommited
'''

def knn(T, x, k, normalized = False, weighted=False, debug=False):
    if normalized:
        T = normalize(T)
    
    # obtain possible neighbors by (distance, class)    
    neighbors = [(distance(x, ex_instance), ex_instance[-1]) for ex_instance in T]
    neighbors.sort() 
    neighbors = neighbors[:k] # and keep K nearest

    results = {}

    if weighted:
        # obtain closest and furthest neighbor distances
        d_1, d_k = neighbors[0][0], neighbors[-1][0]

        for n in neighbors:
            if n[1] not in results:
                results[n[1]] = 0
            results[n[1]] += (d_k - n[0]) / (d_k - d_1) if not d_k == d_1 else 1
    else:
        for n in neighbors:
            if n[1] not in results:
                results[n[1]] = 0
            results[n[1]] += 1

    if debug:
        print(F'n: {neighbors}')
        print(F'buckets {"weighted" if weighted else "unweighted"}: {results}')
    
    
    # the class with most weight classifies X
    return (max(results, key = lambda classifier:results[classifier]))

def distance(x, ex):
    diffs_squared = [(ex[i] - x[i]) ** 2 for i in range(len(x))]
    return math.sqrt(sum(diffs_squared))

'''
    Normalize data using x = (x-MIN)/(MAX - MIN)
    Doesn't normalized discrete values because how?
'''

def normalize(dataset):
    # class is not included
    n_attributes = len(dataset[0]) - 1

    # transpose the dataset to figure out mins and maxs for each attribute 
    attribute_tally = [ [instance[a] for instance in dataset] for a in range(n_attributes) ]

    mins = [ min(a) for a in attribute_tally]
    maxs = [ max(a) for a in attribute_tally]

    # create normalized dataset
    norm_dataset = []
    for instance in dataset:
        norm_instance = []
        for a_i in range(n_attributes):
            norm_a = 1 # for the case that MIN = MAX (avoid division by zero)
            if mins[a_i] != maxs[a_i]:
                norm_a = (instance[a_i] - mins[a_i]) / (maxs[a_i]- mins[a_i])
            norm_instance.append(norm_a)

        norm_instance.append(instance[-1])  # append class
        norm_dataset.append(norm_instance)

    return norm_dataset

# convert strings into types
def parse(a_vector):
    new_vector = []
    for a in a_vector:
        try:
            converted_a = float(a)
            new_vector.append(int(converted_a) if converted_a.is_integer() else converted_a) 
        except ValueError:
            new_vector.append(a)    # this value must be a discrete attribute
    return new_vector
    
# run some parsing tests
if __name__ == "__main__":
    # testing distance function
    x = [2, 4, 2]
    distance_test = [[1, 3, 1, 'classA'], [3, 5, 2, 'ClassB'], [3, 2, 2, 'classC'], [5, 2, 3, 'classD']]
    distance_answers = [math.sqrt(3), math.sqrt(2), math.sqrt(5), math.sqrt(14)]
    distance_results = [distance(x, t) for t in distance_test]

    print('CHECKING distance function...')
    for x, y in zip(distance_results, distance_answers):
        if x != y:
            print('Distance function failed! got ', x, ' when it was supposed to get ', y)

    print('CHECKING parser...')
    unparsed_data = ['1,1,2,a,2.3', '3,4,5,1,k,3.0', '2,a,cb,34,r,3.4']
    unparsed_data = [s.split(',') for s in unparsed_data]

    parsed_results = [parse(a_vector) for a_vector in unparsed_data]
    parsed_answers = [[1, 1, 2, 'a', 2.3], [3, 4, 5, 1, 'k', 3], [2, 'a', 'cb', 34, 'r', 3.4]]
    for x, y in zip(parsed_results, parsed_answers):
        if x != y:
            print('Parser failed! got ', x, ' instead of ', y)
    
    print('CHECKING normalizer...')
    # TODO ^^ THAT

    
    