import numpy as np
import matplotlib.pyplot as plt

data_name1 = 'data1.txt'
data_name2 = 'data2.txt'


def read_data(data_file):
    data = []
    split_string = ' '
    if data_file == 'data1.txt':
        split_string = '\t'
    elif data_file == 'data2.txt':
        split_string = ' '
    with open(data_file) as file:
        for line in file.readlines():
            data_line = np.array(line.split(split_string), dtype='float')
            data.append(data_line)
    data = np.array(data)
    return data


def k_means(data, centers, print_info=False):
    '''
    K_means algorithm.
    :param data: The data that needs to be clustered.
    :param centers: The centers of cluster.
    :return: The cluster result and centers.
    '''
    N = data.shape[0]
    class_num = centers.shape[0]
    standard_ans = np.zeros(N)
    per_class = int(N / class_num)
    for i in range(class_num):
        standard_ans[i * per_class:(i + 1) * per_class] = i

    for step in range(10000):
        # caculate the distance between cluster centers and data
        dis = np.zeros([data.shape[0], class_num])
        for i in range(class_num):
            data_norm2 = np.linalg.norm(data - centers[i], ord=2, axis=1)
            dis[:, i] = data_norm2

        # find the min index of distance matrix
        result = np.argmin(dis, axis=1)
        centers_tmp = np.zeros(centers.shape, dtype='float')
        for i in range(class_num):
            class_inx = np.argwhere(result == i)
            if class_inx.all():
                centers_tmp[i] = np.mean(data[class_inx], axis=0)
        error_matrix = (standard_ans - result)
        error_matrix[error_matrix < 0] = 1
        error_matrix[error_matrix > 0] = 1
        right_num = error_matrix.sum()
        acc = (N - right_num) / N
        if print_info:
            print("Step %d , acc is : %f" % (step, acc))
        if abs(centers - centers_tmp).sum() == 0:
            break
        else:
            centers = centers_tmp
    return result, centers, acc


def spectral_cluster(data, k=10, type='Ng', sim='classical', sigma=5, cluster_num=2):
    N = data.shape[0]
    W = np.zeros([N, N])
    for i in range(N):
        dis = np.linalg.norm(data - data[i], ord=2, axis=1)
        dis = np.delete(dis, 0)
        k_idx = np.argsort(dis)[: k]
        if sim == 'classical':
            W[i, k_idx] = 1
        else:
            W[i, k_idx] = np.exp(-(dis[k_idx] ** 2) / (2 * (sigma ** 2)))
    if sim == 'classical':
        D = k * np.eye(N)
        # W = (W.T + W) / 2
        L = D - W
    else:
        D_diag = np.sum(W, axis=0)
        D = np.diag(D_diag)
        W = (W.T + W) / 2
        L = D - W
    if type == 'Ng':
        L_sys = ((D * (k ** (-0.5))).dot(L)).dot(D * (k ** (-0.5)))
        vals, vecs = np.linalg.eig(L_sys)
        indx = np.argsort(vals)[:cluster_num]
        U = vecs[:, indx]
        T = U / np.sqrt(np.sum(U ** 2, axis=0))
        centers = np.array([T[60], T[150]])
        result, centers, acc = k_means(T, centers)
    return result, centers, acc


if __name__ == '__main__':
    # data1 = read_data(data_name1)
    #######True clusters#######
    # plt.figure()
    # plt.scatter(data1[:200, 0], data1[:200, 1], color='blue', label='Class 1')
    # plt.scatter(data1[200:400, 0], data1[200:400, 1], color='red', label='Class 2')
    # plt.scatter(data1[400:600, 0], data1[400:600, 1], color='green', label='Class 3')
    # plt.scatter(data1[600:800, 0], data1[600:800, 1], color='yellow', label='Class 4')
    # plt.scatter(data1[800:1000, 0], data1[800:1000, 1], color='purple', label='Class 5')
    # plt.legend()
    #######K-means#######
    # orig_clusters = np.array([data1[150], data1[350], data1[550], data1[750], data1[950]], dtype='float')
    # result, centers, acc = k_means(data1, centers=orig_clusters)
    # plt.figure()
    # colors = ['blue', 'red', 'green', 'yellow', 'purple']
    # classes = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    # for i in range(5):
    #     class_inx = np.argwhere(result == i)
    #     plt.scatter(data1[class_inx, 0], data1[class_inx, 1], color=colors[i], label=classes[i])
    # plt.legend()
    # centers_mean = np.array([[1, -1], [5.5, -4.5], [1, 4], [6, 4.5], [9, 0]])
    # error = abs(orig_clusters - centers_mean).sum()
    # print("Error between k-means centers and true centers is %f" % error)
    # print(abs(centers - centers_mean))
    #
    ##########True Clusters##########
    plt.figure()
    data2 = read_data(data_name2)
    plt.scatter(data2[:100, 0], data2[:100, 1], color='blue', label='Class 1')
    plt.scatter(data2[100:200, 0], data2[100:200, 1], color='red', label='Class 2')
    plt.legend()
    # np.random.shuffle(data2)
    # orig_clusters2=np.array([data2[50], data2[150]])
    # result, centers = k_means(data2, centers=orig_clusters2)
    # result, centers, acc = spectral_cluster(data2, sim='others')
    # colors = ['green', 'yellow']
    # classes = ['class 1', 'class 2']
    # plt.figure()
    # for i in range(2):
    #     class_inx = np.argwhere(result == i)
    #     plt.scatter(data2[class_inx, 0], data2[class_inx, 1], color=colors[i], label=classes[i])
    # plt.legend()
    # plt.show()

    ########分析谱聚类参数的影响#######
    sigmas = range(1, 20)
    ks = range(3, 31)
    accuracies = np.zeros([len(sigmas), len(ks)])
    for sigma in sigmas:
        for k in ks:
            result, centers, acc = spectral_cluster(data2, sim='others' ,sigma=sigma, k=k)
            accuracies[sigma - 1, k - 3] = acc
    plt.figure()
    sigma = sigmas[5]
    plt.plot(ks, accuracies[sigma - 1], marker='o', mfc='#4F94CD', label='sigma=5', color='blue', linewidth=2)
    plt.xlim((3, 31))
    plt.ylim((0.2, 1.5))
    plt.xlabel('Different k')
    plt.ylabel('Accuracies')
    plt.legend()
    plt.figure()
    k = 10
    plt.plot(sigmas, accuracies[:, k - 3], marker='o', mfc='#EE6363', label='k=10', color='red', linewidth=2)
    plt.xlim((1, 20))
    plt.ylim((0.2, 1.5))
    plt.xlabel('Different sigma')
    plt.ylabel('Accuracies')
    plt.legend()
    plt.show()
