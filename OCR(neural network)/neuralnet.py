import numpy as np
import argparse
import logging
import sys

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')

def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """
    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0] = 1.0 #add bias terms
    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)



def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))
    # Implement random initialization here
    return np.insert(np.random.uniform(-0.1, 0.1, [shape[0],shape[1]-1]), 0, 0, axis=1)
    raise NotImplementedError


def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape)
    raise NotImplementedError

class NN(object):
    def __init__(self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size, X_tr, y_tr, X_te, y_te):
        """
        Initialization
        :param lr: learning rate
        :param n_epoch: number of training epochs
        :param weight_init_fn: weight initialization function
        :param input_size: number of units in the input layer
        :param hidden_size: number of units in the hidden layer
        :param output_size: number of units in the output layer
        """
        self.lr = lr
        self.n_epoch = n_epoch
        self.weight_init_fn = weight_init_fn
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size

        # initialize weights and biases for the models
        # M+1=n_input, D = n_hidden, K = n_output
        # w1 = Dx(M+1), w2 = Kx(D+1)
        self.w1 = weight_init_fn([self.n_hidden, self.n_input])
        self.w2 = weight_init_fn([self.n_output, self.n_hidden+1])

        # initialize parameters for adagrad
        # self.epsilon =
        # self.grad_sum_w1 =
        # self.grad_sum_w2 =
        self.epsilon = 1e-5
        # feel free to add additional attributes
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_te = X_te
        self.y_te = y_te
        self.label = sorted(set(y_tr))
        self.z = []


def print_weights(nn):
    """
    An example of how to use logging to print out debugging infos.

    Note that we use the debug logging level -- if we use a higher logging
    level, we will log things with the default logging configuration,
    causing potential slowdowns.

    Note that we log NumPy matrices on separate lines -- if we do not do this,
    the arrays will be turned into strings even when our logging is set to
    ignore debug, causing potential massive slowdowns.
    :param nn: your model
    :return:
    """
    logging.debug(f"shape of w1: {nn.w1.shape}")
    logging.debug(nn.w1)
    logging.debug(f"shape of w2: {nn.w2.shape}")
    logging.debug(nn.w2)


def forward(X, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data
    :param nn: neural network class
    :return: output probability
    """
    # a: w1 = Dx(M+1), X = (M+1) x L => DxL
    a = np.matmul(nn.w1, np.transpose(X))
    # z = DxL
    z = 1 / (1 + np.exp(-a))
    # z = (D+1)xL
    z = np.reshape(np.insert(z, 0, 1, axis=0), (len(z) + 1, len(z[0])))
    nn.z = z
    # b : w2 = Kx(D+1), z = (D+1)xL => KxL=> LxK
    b = np.transpose(np.matmul(nn.w2, z))
    # y_hat: LxK
    y_hat = np.exp(b)/np.reshape(np.sum(np.exp(b), axis=1), (b.shape[0],1))
    return y_hat
    raise NotImplementedError


def backward(X, y, y_hat, nn):
    """
    Neural network backward computation.
    Follow the pseudocode!
    :param X: input data
    :param y: label
    :param y_hat: prediction
    :param nn: neural network class
    :return:
    d_w1: gradients for w1
    d_w2: gradients for w2
    """
    # D_crossentropy, 1-1xK = 1xK
    gb = np.sum(y)*y_hat-y
    # D_linear,
    # d_w2: (Kx1)@(1x(D+1))= Kx(D+1)
    d_w2 = np.matmul(np.transpose(gb), np.transpose(nn.z))
    # gz: (1xK)@(Kx(D))= 1x(D)
    gz = np.matmul(gb, nn.w2[:, 1:])
    # D_sigmoid, 1xD * ((1xD)*(1xD)) = 1xD
    z = np.transpose(nn.z[1:])
    ga = gz *((1 - z) * z)
    # D_linear,
    # d_w1: (Dx1) @ ( 1x (M+1)) = Dx(M+1)
    d_w1 = np.matmul(np.transpose(ga), X)
    # gx: 1xD @ (Dx(M+1)) = 1x(M+1)
    # gx = np.matmul(ga, nn.w1)
    return d_w1, d_w2
    raise NotImplementedError


def test(X, X1, nn):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    y = forward(X, nn)
    y_tr_label = np.argmax(y, axis=1)
    error_tr = np.mean(nn.y_tr != y_tr_label)
    y1 = forward(X1, nn)
    y_te_label = np.argmax(y1, axis=1)
    error_te = np.mean(nn.y_te != y_te_label)
    return y_tr_label, y_te_label, error_tr, error_te
    raise NotImplementedError

def train(X_tr, y_tr, nn, X_te, y_te):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param nn: neural network class
    :param X_te: test data
    :param y_te: test label
    """
    st1 = np.zeros((nn.n_hidden, nn.n_input))
    st2 = np.zeros((nn.n_output, nn.n_hidden + 1))
    # initial cross entropy
    JD = []
    JDt = []
    # loop for n_epoch
    for epoch in range(nn.n_epoch):
        # shuffle data
        X, y = shuffle(X_tr, y_tr, epoch)
        #X, y = X_tr, y_tr
        #y_pred = np.zeros(np.shape(y_tr))
        for i in range(len(X)):
            Xi, yi = X[i].reshape((1, len(X[i]))), y[i]
            # Compute neural network layer
            y_hat = forward(Xi, nn)
            #cross = -np.dot(np.log(y_hat), yi.reshape(len(yi), 1))
            # Compute gradients via backprop, d_w1: (Dx1) @ ( 1x (M+1)) = Dx(M+1), d_w2: (Kx1)@(1x(D+1))= Kx(D+1)
            d_w1, d_w2 = backward(Xi, yi, y_hat, nn)
            # Adagrad updates
            st1 += d_w1 * d_w1
            st2 += d_w2 * d_w2
            # Update parameter
            nn.w1 -= (nn.lr / np.sqrt(st1 + nn.epsilon)) * d_w1
            nn.w2 -= (nn.lr / np.sqrt(st2 + nn.epsilon)) * d_w2
        # Evaluate training mean cross-entropy
        y_pred = forward(X, nn)
        JD.append(-np.average(np.diag(np.dot(y, np.transpose(np.log(y_pred))))))
        yt_pred = forward(X_te, nn)
        JDt.append(-np.average(np.diag(np.dot(y_te, np.transpose(np.log(yt_pred))))))
    return nn.w1, nn.w2, JD, JDt
    raise NotImplementedError

if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    # Note: You can access arguments like learning rate with args.learning_rate
    X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics, n_epochs, n_hid, init_flag, lr = args2data(args)
    # initialize training / test data and labels
    # my_nn = NN()
    if init_flag == 1:
        #len(set(y_tr))
        my_nn = NN(lr, n_epochs, random_init, len(X_tr[0]), n_hid, 10, X_tr, y_tr, X_te, y_te)
    else:
        my_nn = NN(lr, n_epochs, zero_init, len(X_tr[0]), n_hid, 10, X_tr, y_tr, X_te, y_te)
    # modify y
    y_tr_mod = np.zeros((len(y_tr), my_nn.n_output))
    for i, index in enumerate(y_tr):
        y_tr_mod[i, index] = 1
    y_te_mod = np.zeros((len(y_te), my_nn.n_output))
    for i, index in enumerate(y_te):
        y_te_mod[i, index] = 1

    # train model
    w1, w2, JD, JDt = train(X_tr, y_tr_mod, my_nn, X_te, y_te_mod)
    # test model and get predicted labels and errors
    y_tr_label, y_te_label,  error_tr, error_te = test(X_tr, X_te, my_nn)
    # write predicted label and error into file
    file1 = open(out_metrics, "w")
    for epoch in range(1, my_nn.n_epoch + 1):
        s = 'epoch=' + str(epoch) + ' crossentropy(train): ' + str(JD[epoch - 1])
        file1.write(s + '\n')
        s = 'epoch=' + str(epoch) + ' crossentropy(validation): ' + str(JDt[epoch - 1])
        file1.write(s + '\n')
    file1.write('error(train): ' + str(error_tr) + '\n')
    file1.write('error(validation): ' + str(error_te))
    file1.close()
    np.savetxt(out_tr, y_tr_label, fmt="%s")
    np.savetxt(out_te, y_te_label, fmt="%s")
