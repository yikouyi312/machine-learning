import csv
import sys
import numpy as np

def load_tsv_dataset(file):
    # read formatted data
    data = []
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            data.append((np.array(row)).astype(np.float64))
    return np.array(data)

def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def loglikelihood(theta, X, y, N):
    # 1/N times the negative conditional log-likelihood
    # N: row size of data, M: column size of data
    # theta: (M+1,1), X:(N,M+1), y:(1,N)
    return 1/N*(-np.dot(y(np.matmul(X, theta)))+np.sum(np.log(1 + np.exp(np.matmul(X, theta)))))

def gradient(theta, X, y, N):
    # GD, gradient, 1/N times the negative conditional log-likelihood
    # N: row size of data, M: column size of data
    # theta: (M+1), X:(N,M+1), y:(1,N)
    return 1/N*(-np.matmul(X, y - sigmoid(np.matmul(X, theta))))

def igradient(theta, X, y, i):
    # SGD, i-th data point
    # N: row size of data, M: column size of data
    # theta: (M+1,), X:(N,M+1), y:(1,N)
    # theta[0]: intercept
    return -X[i]*(y[i] - sigmoid(np.matmul(X[i], theta)))

def train(theta, X, y, num_epoch, learning_rate):
    # TODO: Implement `train` using vectorization
    train_theta = theta
    for i in range(num_epoch):
        SGD = np.arange(np.shape(X)[0])
        #np.random.shuffle(SGD)
        for j in range(np.shape(X)[0]):
            ith = SGD[j]
            train_theta -= learning_rate * igradient(theta, X, y, ith)
    return train_theta


def predict(theta, X):
    # TODO: Implement `predict` using vectorization
    p1 = sigmoid(np.matmul(X, theta))
    return np.intc(p1 > 0.5)

def compute_error(y_pred, y):
    # TODO: Implement `compute_error` using vectorization
   # y_pred = y_pred.reshape(np.shape(y_pred)[0], 1)
    error_rate = np.mean(y != y_pred)
    return error_rate

# train_input = 'smalloutput/model1_formatted_train.tsv'
# test_input = 'smalloutput/model1_formatted_test.tsv'
# train_output = 'train.label'
# test_output = 'test.label'
# metric_out = 'metric.txt'
# num_epoch = 500
# learning_rate = 0.00003


def logisticregression(train_input, validation_input, test_input, train_output,
                      test_output, metric_out, num_epoch, learning_rate):
    # read data
    train_data = load_tsv_dataset(train_input)
    #valid_data = load_tsv_dataset(validation_input)
    test_data = load_tsv_dataset(test_input)
    # train
    #y = train_data[:, 0].reshape((np.shape(train_data)[0], 1))
    y = train_data[:, 0]
    X = train_data[:, 1:]
    X = np.insert(X, 0, 1, axis=1)  # fold intercept into theta
    theta = np.zeros((np.shape(X)[1]))  #initial theta
    train_theta = train(theta, X, y, num_epoch, learning_rate)
    y_pred = predict(train_theta, X)  #predict train data
    train_error = compute_error(y_pred, y)
    np.savetxt(train_output, y_pred, fmt="%s")
    #predict
    #y = test_data[:, 0].reshape((np.shape(test_data)[0], 1))
    y = test_data[:, 0]
    X = test_data[:, 1:]
    X = np.insert(X, 0, 1, axis=1) # fold intercept into theta
    y_pred = predict(train_theta, X)
    test_error = compute_error(y_pred, y)
    np.savetxt(test_output, y_pred, fmt="%s")
    np.savetxt(metric_out,
               np.array([['error(train): %f' % train_error], ['error(test): %f' % test_error]]), fmt='%s')

if __name__ == "__main__":
    # arguments
    train_input: str = sys.argv[1]
    validation_input: str = sys.argv[2]
    test_input: str = sys.argv[3]
    train_output: str = sys.argv[4]
    test_output: str = sys.argv[5]
    metric_out: str = sys.argv[6]
    num_epoch: int = int(sys.argv[7])
    learning_rate: float = float(sys.argv[8])
    logisticregression(train_input, validation_input, test_input, train_output,
                       test_output, metric_out, num_epoch, learning_rate)














