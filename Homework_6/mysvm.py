from svmutil import *
from svm import *
import os
import numpy as np
import gzip


def load_data(data_file):
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']
    paths = []
    for file in files:
        paths.append(os.path.join(data_file, file))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(
            lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(
            lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    return (x_train, y_train), (x_test, y_test)


def make_data_input(default=[0, 1]):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    (train_images, train_labels), (test_images, test_labels) = load_data('MNIST/')
    for k, image in enumerate(train_images):
        if int(train_labels[k]) in default:
            image_flatten = image.reshape(28 * 28, 1)
            x_img = {i: int(image_flatten[i]) for i in range(28 * 28)}
            x_train.append(x_img)
            y_train.append(int(train_labels[k]))

    for k, image in enumerate(test_images):
        if int(test_labels[k]) in default:
            image_flatten = image.reshape(28 * 28, 1)
            x_img = {i: int(image_flatten[i]) for i in range(28 * 28)}
            x_test.append(x_img)
            y_test.append(int(test_labels[k]))
    return (x_train, y_train, x_test, y_test)


def make_data_file(default=[0, 1]):
    (train_images, train_labels), (test_images, test_labels) = load_data('MNIST/')
    with open('train_mnist.txt', 'w') as file:
        for k, image in enumerate(train_images):
            if int(train_labels[k]) in default:
                image_flatten = image.reshape(28 * 28, 1)
                x_img = str(train_labels[k]) + ' '
                for i in range(1, 28 * 28 + 1):
                    x_img += str(i) + ':' + str(image_flatten[i - 1][0]) + ' '
                x_img.strip(' ')
                x_img += '\n'
                file.write(x_img)

    with open('test_mnist.txt', 'w') as file:
        for k, image in enumerate(test_images):
            if int(train_labels[k]) in default:
                image_flatten = image.reshape(28 * 28, 1)
                x_img = str(test_labels[k]) + ' '
                for i in range(1, 28 * 28 + 1):
                    x_img += str(i) + ':' + str(image_flatten[i - 1][0]) + ' '
                x_img.strip(' ')
                x_img += '\n'
                file.write(x_img)


if __name__ == '__main__':
    #######Make dataset#######
    # make_data_file()
    x_train, y_train, x_test, y_test = make_data_input()

    cs = [10, 100, 200, 500, 1000, 10000]
    gammas = [0.00001, 0.000001, 0.0000001, 0.00000001]
    accs = np.zeros([len(cs), len(gammas)], dtype='float')
    prob = svm_problem(y_train, x_train)
    for i, c in enumerate(cs):
        for j, gamma in enumerate(gammas):
            parameter = '-t 2 -c %d -g %f' % (c, gamma)
            param = svm_parameter(parameter)
            m = svm_train(prob, param)
            p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
            accs[i, j] = p_acc[0]
    print(accs)