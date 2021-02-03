import numpy as np
import matplotlib.pyplot as plt


def load_data(data_path):
    data = []
    with open(data_path) as data_file:
        for line in data_file.readlines():
            line_split = line.strip('\n').split(' ')
            line_split = [float(line_split[i]) for i in range(len(line_split))]
            data.append(line_split)
    return np.array(data)


def batch_perception(lr=0.5, data=None):
    a = np.zeros([3, 1])
    threshold = 1e-9
    data[:, 2] = 1
    data[10:, :] = -data[10:, :]
    iter = 0
    for i in range(1000000):
        result = a.T.dot(data.T)
        _, inx = np.where(result <= 0)
        increments = np.sum(lr * data[inx], axis=0)
        a += increments.reshape([3, 1])
        iter += 1
        if abs(increments.sum()) < threshold:
            print('total iter:%d' % iter)
            return a


def Ho_Kashyap(lr=0.01, data=None, threshold=1e-10):
    a = np.ones([3, 1])
    b = np.ones([len(data), 1])
    data[:, 2] = 1
    data[10:, :] = -data[10:, :]
    for i in range(300000):
        error = data.dot(a) - b
        error_plus = 0.5 * (error + np.abs(error))
        b = b + 2 * lr * error_plus
        a = np.linalg.pinv(data).dot(b)
        if i % 100 == 0:
            print('iter:%d, the error is %f' % (i, error.sum()))
        if abs(error).sum() < threshold:
            print(error)
            return a
    print(error)
    return a


def multi_class_mse(epsilon=0.01, data=None, label=None):
    W = np.linalg.inv(data.dot(data.T) + epsilon).dot(data).dot(label.T)
    return W


if __name__ == '__main__':
    data = load_data('data.txt')
    # show the data
    plt.figure()
    plt.scatter(data[:10, 0], data[:10, 1], marker='o', label='1', color=(0.8, 0., 0.), s=70)
    plt.scatter(data[10:20, 0], data[10:20, 1], marker='x', label='2', color=(0., 0., 0.4), s=70)
    plt.scatter(data[20:30, 0], data[20:30, 1], marker='+', label='3', color=(0., 0.8, 0.), s=80)
    plt.scatter(data[30:40, 0], data[30:40, 1], marker='^', label='4', color=(0.4, 0.3, 0.2), s=70)
    plt.legend(loc='upper left')

    #####batch_perception#####
    # train data of w1 and w2
    data_temp = np.zeros([20, 3])
    data_temp[:10] = data[:10]
    data_temp[10:20] = data[10:20]
    a = batch_perception(data=data_temp)
    a = a / a[1]
    a_x = np.linspace(-7.5, 10, 10)
    a_y = -a_x * a[0] - a[2]
    plt.figure()
    plt.scatter(data[:10, 0], data[:10, 1], marker='o', label='1', color=(0.8, 0., 0.), s=70)
    plt.scatter(data[10:20, 0], data[10:20, 1], marker='x', label='2', color=(0., 0., 0.4), s=70)
    plt.plot(a_x, a_y, '-', label='Boundary')
    plt.legend(loc='upper left')

    # train data of w2 and w3
    data_temp = np.zeros([20, 3])
    data_temp[:10] = data[10:20]
    data_temp[10:20] = data[20:30]
    a = batch_perception(data=data_temp)
    a = a / a[1]
    a_x = np.linspace(-7.5, 10, 10)
    a_y = -a_x * a[0] - a[2]
    plt.figure()
    plt.scatter(data[10:20, 0], data[10:20, 1], marker='x', label='2', color=(0., 0., 0.4), s=70)
    plt.scatter(data[20:30, 0], data[20:30, 1], marker='+', label='3', color=(0., 0.8, 0.), s=80)
    plt.plot(a_x, a_y, '-', label='Boundary')
    plt.legend(loc='upper right')

    #####Ho_Kashyap#####
    # train data of w1 and w3
    data_temp = np.zeros([20, 3])
    data_temp[:10] = data[:10]
    data_temp[10:20] = data[20:30]
    a = Ho_Kashyap(data=data_temp, threshold=1)
    a = a / a[1]
    a_x = np.linspace(-7.5, 10, 10)
    a_y = - a_x * a[0] - a[2]
    plt.figure()
    plt.scatter(data[:10, 0], data[:10, 1], marker='o', label='1', color=(0.8, 0., 0.), s=70)
    plt.scatter(data[20:30, 0], data[20:30, 1], marker='+', label='3', color=(0., 0.8, 0.), s=80)
    plt.plot(a_x, a_y, '-', label='Boundary')
    plt.legend(loc='upper left')

    # train data of w2 and w4
    data_temp = np.zeros([20, 3])
    data_temp[:10] = data[10:20]
    data_temp[10:20] = data[30:40]
    a = Ho_Kashyap(data=data_temp)
    a = a / a[1]
    a_x = np.linspace(-10, 8, 10)
    a_y = - a_x * a[0] - a[2]
    plt.figure()
    plt.scatter(data[10:20, 0], data[10:20, 1], marker='x', label='2', color=(0., 0., 0.4), s=70)
    plt.scatter(data[30:40, 0], data[30:40, 1], marker='^', label='4', color=(0.4, 0.3, 0.2), s=80)
    plt.plot(a_x, a_y, '-', label='Boundary')
    plt.legend(loc='upper left')

    #####Multi-Classification######
    train_data = np.zeros([3, 32])
    train_data[:, :8] = data[:8].T
    train_data[:, 8:16] = data[10:18].T
    train_data[:, 16:24] = data[20:28].T
    train_data[:, 24:32] = data[30:38].T
    train_data[2, :] = 1

    train_label = np.zeros([4, 32])
    train_label[0, :8] = 1
    train_label[1, 8:16] = 1
    train_label[2, 16:24] = 1
    train_label[3, 24:32] = 1

    W = multi_class_mse(data=train_data, label=train_label)

    test_data = np.zeros([3, 8])
    test_data[:, :2] = data[8:10].T
    test_data[:, 2:4] = data[18:20].T
    test_data[:, 4:6] = data[28:30].T
    test_data[:, 6:8] = data[38:40].T
    test_data[2, :] = 1

    test_label = np.zeros([4, 8])
    test_label[0, :2] = 1
    test_label[1, 2:4] = 1
    test_label[2, 4:6] = 1
    test_label[3, 6:8] = 1

    predict = W.T.dot(test_data)
    inx = np.argmax(predict, axis=0)
    cnt = 0
    for i in range(8):
        if test_label[inx[i], i] == 1:
            cnt += 1
    acc = cnt / test_label.shape[1]
    print('The accuracy is: %f' % acc)

    plt.show()
