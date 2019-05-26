from perceptron import Perceptron
from svm import Svm
from pa import Pa
import numpy as np
import sys


def read_labels(file_name):
    """
    read labels
    each label is in a new line
    rows according rows of points
    """
    return list(np.genfromtxt(file_name, delimiter=','))


def read_points(file_name):
    """
    read points
    rows according rows of labels
    convert 'M', 'F' and 'I' to three binary fields (one hot)
    """
    lines = np.genfromtxt(file_name, delimiter=',', dtype='str')
    points = []
    for i in range(len(lines)):
        # one hot
        one_hot = np.array([0.0, 0.0, 0.0])
        gender = lines[i][0]
        if gender == 'M':
            one_hot[0] = 1.0
        elif gender == 'F':
            one_hot[1] = 1.0
        elif gender == 'I':
            one_hot[2] = 1.0
        point = np.delete(lines[i], 0).astype(np.float)
        point = np.concatenate((one_hot, point))
        points.append(list(point))
    return points


def printing_format(perceptron, svm, pa):
    """
    print labels of three algos according format
    """
    for i in range(len(perceptron)):
        line = "perceptron: " + str(perceptron[i])
        line += ", svm: " + str(svm[i])
        line += ", pa: " + str(pa[i])
        print(line)


def get_mean_std(points):
    """
    get mean and std of features
    """
    means, stds, col = [], [], []
    # iterate over features (cols)
    for i in range(len(points[0])):
        # iterate over points (rows)
        for point in points:
            col.append(point[i])
        # add mean and std of col
        means.append(np.mean(np.array(col)))
        stds.append(np.std(np.array(col)))
        col = []
    return means, stds


def get_min_max(points):
    """
    get min and max of features
    """
    mins, maxs, col = [], [], []
    # iterate over features (cols)
    for i in range(len(points[0])):
        # iterate over points (rows)
        for point in points:
            col.append(point[i])
        # add min and max of col
        mins.append(np.min(np.array(col)))
        maxs.append(np.max(np.array(col)))
        col = []
    return mins, maxs


def apply_mean_std_bias(points, means, stds):
    """
    apply normalize according mean and std
    and add bias
    """
    # iterate over features (cols)
    for i in range(len(points[0])):
        # iterate over points (rows)
        for j in range(len(points)):
            points[j][i] = (points[j][i] - means[i]) / stds[i]
    # add bias
    for i in range(len(points)):
        points[i] = np.concatenate((points[i], np.array([1])))
    return points


def apply_min_max_bias(points, mins, maxs):
    """
    apply normalize according min and max
    and add bias
    """
    # iterate over features (cols)
    for i in range(len(points[0])):
        # iterate over points (rows)
        for j in range(len(points)):
            points[j][i] = (points[j][i] - mins[i]) / (maxs[i] - mins[i])
    # add bias
    for i in range(len(points)):
        points[i] = np.concatenate((points[i], np.array([1])))
    return points


def main():
    # files names
    x_train_name = sys.argv[1]
    y_train_name = sys.argv[2]
    x_test_name = sys.argv[3]
    # y_test_name = "test_y.txt" # todo

    # read data
    train_x = read_points(x_train_name)
    train_y = read_labels(y_train_name)
    test_x = read_points(x_test_name)
    # test_y = read_labels(y_test_name) # todo

    try:
        # try to normalize according mean and std
        mean, std = get_mean_std(train_x)
        train_x = apply_mean_std_bias(train_x, mean, std)
        test_x = apply_mean_std_bias(test_x, mean, std)

        # try to normalize according min and max
        # min, max = get_min_max(train_x)
        # train_x = apply_min_max_bias(train_x, min, max)
        # test_x = apply_min_max_bias(test_x, min, max)
    except:
        nothing = "nothing"

    # prepare set
    train_set = list(zip(train_x, train_y))
    # test_set = list(zip(test_x, test_y)) # todo

    # choose better (probably) trained
    max_accs, best_algos = [0, 0, 0], [None, None, None]
    num_tries = 10
    for i in range(num_tries):
        try:
            algos = [Perceptron(list(train_set), list(test_x), 5, 0.01),
                     Svm(list(train_set), list(test_x), 10, 0.01, 0.001),
                     Pa(list(train_set), list(test_x), 10)]

            accs, vaccs = [], []
            for algo in algos:
                algo.train()
                accs.append(algo.validate(train_set))
                # vaccs.append(algo.k_cross_validation()) # todo
            # print(accs)
            # print(vaccs)
            # print("\n")

            # take algo of better acc
            for i in range(len(algos)):
                if accs[i] > max_accs[i]:
                    best_algos[i], max_accs[i] = algos[i], accs[i]
        except:
            nothing = "nothing"

    # print([best_algos[0].validate(train_set),
    #        best_algos[1].validate(train_set),
    #        best_algos[2].validate(train_set)])
    # print([best_algos[0].validate(test_set),
    #        best_algos[1].validate(test_set),
    #        best_algos[2].validate(test_set)])

    for best_alg in best_algos:
        best_alg.predict_all()

    printing_format(best_algos[0].test_y, best_algos[1].test_y, best_algos[2].test_y)


if __name__ == '__main__':
    main()
