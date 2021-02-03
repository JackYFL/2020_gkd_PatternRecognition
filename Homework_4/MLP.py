import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def visualize_3D_data(train_data):
    mpl.rcParams['legend.fontsize'] = 20  # mpl模块载入的时候加载配置信息存储在rcParams变量中，rc_params_from_file()函数从文件加载配置信息
    font = {
        'color': 'b',
        'style': 'oblique',
        'size': 20,
        'weight': 'bold'
    }
    fig = plt.figure(figsize=(16, 12))  # 参数为图片大小
    ax = fig.gca(projection='3d')  # get current axes，且坐标轴是3d的
    # ax.set_aspect('equal')  # 坐标轴间比例一致

    tmp = 0
    for c, m, l in [('r', 'o', 'class 1'),
                    ('g', '*', 'class 2'),
                    ('b', '^', 'class 3')]:
        x = train_data[tmp:(tmp + 10), 0]
        y = train_data[tmp:(tmp + 10), 1]
        z = train_data[tmp:(tmp + 10), 2]
        tmp += 10
        ax.scatter(x, y, z, c=c, marker=m, label=l, s=80)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    # 设置坐标轴范围
    # ax.set_xlim(-30, 30)
    # ax.set_ylim(-30, 30)
    # ax.set_zlim(-30, 30)
    # ax.set_title("Scatter plot", alpha=0.6, color="b", size=25, weight='bold', backgroundcolor="y")  # 子图的title
    ax.legend(loc="upper left")  # legend的位置左上

    plt.show()


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x)  # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension

    return grad


def load_data(data_path):
    data = []
    with open(data_path) as data_file:
        for line in data_file.readlines():
            line_split = line.strip('\n').split(' ')
            line_split = [float(line_split[i]) for i in range(len(line_split))]
            data.append(line_split)
    return np.array(data)


class ThreeLayerNet(object):
    """
    A three layer perceptron neural network.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1):
        """
        Initialize the model.
        :param input_size: The dimension D of the input data
        :param hidden_size: Hidden layer size.
        :param output_size: The number of classes C.
        :param std: The parameters that initialize the weights.
        W1: (D, h)
        b1: (h, )
        W2: (h, C)
        b2: (C, )
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, x, y=None, reg=0.0):
        """
        Compute the loss and gradients for a three layer FC.
        :param x: Inpute data of shape (N, D). Each x[i] is a training sample.
        :param y: Vectors of training labels of shape (N, C). y[i] is one hot vector. This parameter is optional;
        if it is not passed then we only return scores, and if it is passed then
        we instead return the loss and gradients.
        :param reg: Regularization strength.
        :return: If y==None, return a matrix scores of shape (N,C) where scores[i, c]
        is the score for class c on input x[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
        samples.
        - grads: Dictionary mapping parameters names to gradients of those parameters
        with respect to the loss function; has the same keys as self.params.
        """
        y = np.eye(3)[y]  # one hot
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # Compute the forward pass
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
        H = x.dot(W1) + b1  # (N ,h)
        H_tanh = np.tanh(H)  # hidden layer 1 tanh activation function
        scores = H_tanh.dot(W2) + b2  # (N, C)
        output = sigmoid(scores)  # (N, C)

        # If the labels are not given, just jump out.
        if y is None:
            return output

        # Compute the loss
        loss = np.sum((output - y) ** 2) / 2

        # backward pass: compute gradients
        grads = {}
        doutput = (output - y)  # (N, C)
        dsigmoid = output * (1 - output) * doutput  # (N, C)
        grads['b2'] = np.sum(dsigmoid, axis=0)  # (, C)
        grads['W2'] = H_tanh.T.dot(dsigmoid)  # (h, C)
        dH_tanh = dsigmoid.dot(W2.T)  # (N, h)
        dH = (1 - H_tanh ** 2) * dH_tanh  # (N, h)
        grads['b1'] = np.sum(dH, axis=0)  # (h,)
        grads['W1'] = x.T.dot(dH)  # (D,h)

        return loss, grads

    def train(self, X, y,
              learning_rate=8e-2,
              epochs=10000,
              batch_size=5, verbose=False):
        """
        Train the net with mini-batch SGD
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N, C) giving training labels with one hot vector
        - learning_rate: Scalar giving learning rate for optimization.
        - epochs: Number of epochs to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """

        num_train = X.shape[0]
        iterations_per_epoch = max(int(num_train / batch_size), 1)

        loss_history = []
        train_acc_history = []

        for epoch in range(epochs):
            for iter in range(iterations_per_epoch):

                # Create a random minibatch of training data and labels, storing
                # them in X_batch and y_batch respectively.
                batch_index = np.random.choice(range(num_train), batch_size, replace=True)
                X_batch = X[batch_index]
                y_batch = y[batch_index]

                # Compute loss and gradients using the current minibatch
                loss, grads = self.loss(X_batch, y=y_batch)
                loss_history.append(loss)

                # Update the parameters of the network
                self.params['W2'] -= grads['W2'] * learning_rate
                self.params['b2'] -= grads['b2'] * learning_rate
                self.params['W1'] -= grads['W1'] * learning_rate
                self.params['b1'] -= grads['b1'] * learning_rate

                if verbose and iter % 2 == 0:
                    print('iteration %d / %d: loss %f' % (iter, epoch, loss))

                if epoch % 100 == 0:
                    # Check accuracy
                    train_acc = (self.predict(X) == y).mean()
                    train_acc_history.append(train_acc)
                    # loss_history.append(loss)
                    # print('epoch %d / %d: loss %f train acc %f' % (epoch, epochs, loss, train_acc))

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
        }

    def predict(self, x):
        """
        Use the net to predict labels for data points
        :param x: training data
        :return:
        """
        H = np.maximum(0, x.dot(self.params['W1']) + self.params['b1'])
        H_tanh = np.tanh(H)
        scores = H_tanh.dot(self.params['W2']) + self.params['b2']
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
        output = sigmoid(scores)
        y_pred = np.argmax(output, axis=1)
        return y_pred


if __name__ == '__main__':
    ####Load data####
    raw_data = load_data('data.txt')
    train_data = raw_data[:, :3]
    train_label = (raw_data[:, 3]).astype(np.int)

    ####Train net####
    N, D = train_data.shape
    net = ThreeLayerNet(input_size=D, hidden_size=50, output_size=3)
    # result = net.train(X=train_data, y=train_label)
    # loss_history = result['loss_history']
    # plt.figure()
    # plt.plot(range(len(loss_history)), loss_history, linewidth=1)
    # plt.xlabel("Step")  # xlabel、ylabel：分别设置X、Y轴的标题文字。
    # plt.ylabel("Loss")
    # plt.show()


    train_restore = {}
    loss_restore = {}
    # for hid_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
    #     net = ThreeLayerNet(input_size=D, hidden_size=50, output_size=3)
    #     result = net.train(X=train_data, y=train_label)
    #     train_restore[hid_num] = result['train_acc_history'][-1]
    #     # loss_restore[hid_num] = result['loss_history'][-1]
    # print(train_restore)
    # print(loss_restore)
    ########
    ####Debug####
    # loss, grads = net.loss(x=train_data, y=train_label)
    # for param_name in grads:
    #     f = lambda W: net.loss(train_data, train_label, reg=0.05)[0]
    #     param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    #     print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

    ####Visualization####
    # visualize_3D_data(train_data)
