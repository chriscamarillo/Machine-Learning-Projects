from os import listdir
from os.path import isfile, join
from perceptron import *
from random import shuffle, seed
from helper import *
import matplotlib
import matplotlib.pyplot as plt

# assume last element is a binary class
def read_data(name, classAt=-1):
    f = open(name, 'r')
    data = []

    for line in f.readlines():
        entry = re.split('[, ]+', line)
        data.append(entry)


    first_class = data[0][classAt]

    # Mark differing class labels with 0's and 1's for the perceptron
    for i, entry in enumerate(data):
        entry[classAt] = 0 if entry[classAt] == first_class else 1
        if classAt != -1:
            class_is = entry.pop(classAt)
            entry.append(class_is)
        
        data[i] = [float(attr) for attr in entry]

    return data

def stratify(dataset):

    train_len = 0.2

    # sort T by classifier (last entry)
    dataset.sort(key = lambda t:t[-1])

    # split T into x lists (x = amount of classes) -> call class_split
    class_split = [[x for x in dataset if x[-1] == c] for c in [0, 1]]
    
    num_class_0 = len(class_split[0])
    num_class_1 = len(class_split[1])
    size = len(dataset)

    print(F'# of Class 0: {num_class_0}\t# of Class 1: {num_class_1}')
    print(F'#C0: {num_class_0 / size * 100}')
    print(F'#C1: {num_class_1 / size * 100}')

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


    shuffle(train)
    shuffle(test)
    return (train, test)

# returns EPOCHS and the number of individual training and 
# testing errors per class
def run_simulation(train, test):
    num_attr = len(train[0]) - 1 # exclude class marker
    p = Perceptron(num_attr)
    epochs, errors_on_epoch = p.train_dataset(train)
    
    # testing time
    # gets the number of errors per class
    test_errors_by_class = p.test_dataset(test)

    # return packed data
    return (epochs, errors_on_epoch, test_errors_by_class, p.weights)

def full_comparison_test(filename):
    # load data, stratify, and run_simulations
    dataset = read_data(filename)
    normalize_dataset(dataset)

    vanilla_train, test = stratify(dataset)
    oversampled_train = oversample(vanilla_train)
    undersampled_train = undersample(vanilla_train)

    # vanilla
    vanilla_results = run_simulation(vanilla_train, test)
    undersampled_results = run_simulation(undersampled_train, test)
    oversampled_results = run_simulation(oversampled_train, test)

    results = (vanilla_results, undersampled_results, oversampled_results)
    plots = (vanilla_train, undersampled_train)

    return results, plots, test

def print_confusion_matrix(test_errors_by_class, test):
    # lets say 0 is POS 1 is NEG
    print('total # of items in test: ', len(test))
    total_1s = sum(t[-1] for t in test)
    total_0s = len(test) - total_1s

    # TODO: powerpoint explanation
    true_0s = test_errors_by_class[0]
    true_1s = test_errors_by_class[1]
    false_0s = test_errors_by_class[2]
    false_1s = test_errors_by_class[3] 

    # ALL ABOUT THE POS
    precision = true_0s / (true_0s + false_0s) if true_0s + false_0s > 0 else 0 # TP / Actual Results   
    recall = true_0s / (true_0s + false_1s) if true_0s + false_1s > 0 else 0 # TP / Predicted Results
    f1_score = 2* (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    print(F'total_0s: {total_0s}\ttotal_1s: {total_1s}\n\n')
    print(F'true_0s: {true_0s}\tfalse_0s: {false_0s}')
    print(F'true_1s: {true_1s}\tfalse_1s: {false_1s}')


    print(F'total # of errors: {false_0s + false_1s}')
    print(F'errors from class 0: {false_1s} class 1: {false_0s}')
    print(F'test error rate: {(false_0s + false_1s) / len(test) * 100}%')
    print(F'precision : {precision}')
    print(F'recall: {recall}')
    print(F'F1 score: {f1_score}')
    print(F'Confusion matrix:')
    print('\tpredicted  \n\t\t0\t1\n')
    print('Actual: 0:'.ljust(12), [true_0s, false_0s])
    print('        1:'.ljust(12), [false_1s, true_1s])


def show_results(results, plots, test, filename):
    # unpack
    (vanilla_results, undersampled_results, oversampled_results) = results
    (vanilla_train, undersampled_train) = plots

    plottable = True if len(vanilla_train[0]) == 3 else False
    print('is it plottable?: ', plottable)
    # undersample make lighter

    fig, (ax_epoch, ax_train, ax_test) = plt.subplots(nrows=1, ncols=3, figsize=(7, 7))
    fig.suptitle('Group 6 Imbalanced (using PLA)')

    # epoch vs error rate graph
    ax_epoch.set_title(F'{filename} TRAIN error rate vs epoch')
    ax_epoch.set_xlabel('epoch (t)')
    ax_epoch.set_ylabel('error rate (%)')

    ax_epoch.plot(np.arange(vanilla_results[0]), 100 * np.array(vanilla_results[1]) / len(vanilla_train), 
                            'b', label='vanilla')

    ax_epoch.plot(np.arange(undersampled_results[0]), 100 * np.array(undersampled_results[1]) / len(undersampled_train),
                             'k', label='undersampled')


    ax_epoch.plot(np.arange(oversampled_results[0]), 100 * np.array(oversampled_results[1]) / len(vanilla_train),
                             'g', label='oversampled')
    ax_epoch.legend()

    if plottable:
        # 2D graph of training data + resulting line from weights
        ax_train.set_title(F'{filename} resulting line from training')
        ax_train.set_xlabel(F'x')
        ax_train.set_ylabel(F'y')

        # scatter vanilla points as faded and undersampled as bold
        for point in vanilla_train:
            ax_train.scatter(point[0], point[1], color='orange' if point[-1] == 1 else 'green', alpha=0.3)
        for point in undersampled_train:
            ax_train.scatter(point[0], point[1], color='orange' if point[-1] == 1 else 'green')
        
        # perceptron lines
        v_weights = vanilla_results[-1]
        u_weights = undersampled_results[-1]
        o_weights = oversampled_results[-1]
        
        v_slope, v_intercept = -(v_weights[0] / v_weights[2]) / (v_weights[0]/v_weights[1]), -v_weights[0] / v_weights[2]
        u_slope, u_intercept = -(u_weights[0] / u_weights[2]) / (u_weights[0]/u_weights[1]), -u_weights[0] / u_weights[2]
        o_slope, o_intercept = -(o_weights[0] / o_weights[2]) / (o_weights[0]/o_weights[1]), -o_weights[0] / o_weights[2]
        print(v_slope, ', ', v_intercept)
        x_vals = np.array(ax_train.get_xlim())
        vy_vals = v_slope * x_vals + v_intercept
        uy_vals = u_intercept + u_slope * x_vals
        oy_vals = o_intercept + o_slope * x_vals
        
        ax_train.plot(x_vals, vy_vals, 'b', label='vanilla')
        ax_train.plot(x_vals, uy_vals, 'k', label='undersampled')
        ax_train.plot(x_vals, oy_vals, 'g', label='oversampled')
        ax_train.legend()
        
        
        # 2D graph of testing data + resulting line from weights    
        ax_test.set_title(F'{filename} test results')
        ax_test.set_xlabel(F'x')
        ax_test.set_ylabel(F'y')

        for point in test:
            ax_test.scatter(point[0], point[1], color='orange' if point[-1] == 1 else 'green')
        
        x_vals = np.array(ax_train.get_xlim())
        vy_vals = v_intercept + v_slope * x_vals
        uy_vals = u_intercept + u_slope * x_vals
        oy_vals = o_intercept + o_slope * x_vals
        
        ax_test.plot(x_vals, vy_vals, 'b', label='vanilla')
        ax_test.plot(x_vals, uy_vals, 'k', label='undersampled')
        ax_test.plot(x_vals, oy_vals, 'g', label='oversampled')
        ax_test.legend()
    
    v_epoch_min_x = vanilla_results[1].index(min(vanilla_results[1]))
    v_epoch_min_y = min(vanilla_results[1]) / len(vanilla_train)
    u_epoch_min_x = undersampled_results[1].index(min(undersampled_results[1]))
    u_epoch_min_y = min(undersampled_results[1]) / len(undersampled_train)
    o_epoch_min_x = oversampled_results[1].index(min(oversampled_results[1]))
    o_epoch_min_y = min(oversampled_results[1]) / len(vanilla_train)
 
    epoch_mins = [(v_epoch_min_x, v_epoch_min_y), (u_epoch_min_x, u_epoch_min_y), (o_epoch_min_x, o_epoch_min_y)]
    epoch_train_set_sizes = [len(vanilla_train), len(undersampled_train), len(vanilla_train)]
    for i, test_type in enumerate(['vanilla', 'undersampled', 'oversampled']):
        print('-'*40)
        print(F'{filename} reached lowest training error rate of {epoch_mins[i][1] * 100} at epoch: {epoch_mins[i][0]}')
        print(F'{filename} {test_type} confusion matrix')
        print_confusion_matrix(results[i][-2], test)

    ax_test.set_xlim(0, 1)
    ax_train.set_xlim(0, 1)
    ax_test.set_ylim(0, 1)
    ax_train.set_ylim(0, 1)
    

    plt.show()

if __name__ == '__main__':
    # load in datasets 
    path = 'datasets'
    files = [f for f in listdir(path) if isfile(join(path, f))]

    for f in files:
        results, plots, test = full_comparison_test(join(path, f))
        show_results(results, plots, test, f)

    