import csv
import matplotlib.pyplot as plt
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
    # theta: (M+1,), X:(N,M+1), y:(N,)
    return 1/N*(-np.dot(y, np.matmul(X, theta))+np.sum(np.log(1 + np.exp(np.matmul(X, theta)))))

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

def train(theta, X, y, num_epoch, learning_rate, X_valid, y_valid):
    # TODO: Implement `train` using vectorization
    train_theta = theta
    SGD = np.arange(np.shape(X)[0])
    for i in range(num_epoch):
        #np.random.shuffle(SGD)
        for j in range(np.shape(X)[0]):
            ith = SGD[j]
            train_theta -= learning_rate * igradient(theta, X, y, ith)
        train_log[i] = loglikelihood(train_theta, X, y, np.shape(X)[0])
        valid_log[i] = loglikelihood(train_theta, X_valid, y_valid, np.shape(X_valid)[0])
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

train_input = 'largeoutput/model1_formatted_train.tsv'
valid_input = 'largeoutput/model1_formatted_valid.tsv'
test_input = 'largeoutput/model1_formatted_test.tsv'
train_output = 'train.label'
test_output = 'test.label'
metric_out = 'model1metric.txt'
num_epoch = 5000
learning_rate = 0.00001

map3_valid = np.loadtxt('model3_val_nll.txt')
map3_valid = map3_valid[:, 1]
#def logisticregression(train_input, validation_input, test_input, train_output,
#                     test_output, metric_out, num_epoch, learning_rate):
# read data
train_data = load_tsv_dataset(train_input)
valid_data = load_tsv_dataset(valid_input)
test_data = load_tsv_dataset(test_input)
# train

y = train_data[:, 0]
X = train_data[:, 1:]
X = np.insert(X, 0, 1, axis=1)  # fold intercept into theta
y_valid = valid_data[:, 0]
X_valid = valid_data[:, 1:]
X_valid = np.insert(X_valid, 0, 1, axis=1)


train_log = np.zeros(num_epoch)
valid_log = np.zeros(num_epoch)
theta = np.zeros((np.shape(X)[1]))
theta = train(theta, X, y, num_epoch, learning_rate, X_valid, y_valid)

valid_log1 = valid_log



train_input = 'largeoutput/model2_formatted_train.tsv'
valid_input = 'largeoutput/model2_formatted_valid.tsv'
test_input = 'largeoutput/model2_formatted_test.tsv'
# read data
train_data = load_tsv_dataset(train_input)
valid_data = load_tsv_dataset(valid_input)
test_data = load_tsv_dataset(test_input)
# train

y = train_data[:, 0]
X = train_data[:, 1:]
X = np.insert(X, 0, 1, axis=1)  # fold intercept into theta
y_valid = valid_data[:, 0]
X_valid = valid_data[:, 1:]
X_valid = np.insert(X_valid, 0, 1, axis=1)

train_log = np.zeros(num_epoch)
valid_log = np.zeros(num_epoch)
theta = np.zeros((np.shape(X)[1]))
theta = train(theta, X, y, num_epoch, learning_rate, X_valid, y_valid)

valid_log2 = valid_log

fig, ax = plt.subplots()
num = list(range(1, num_epoch+1))
blue_line = plt.plot(num, valid_log1, color='blue', label='Model1')
red_line = plt.plot(num, valid_log2, color='red',  label='Model2')
green_line = plt.plot(num, map3_valid, color='green',  label='Model3')
plt.xlabel('number of epochs')
plt.ylabel('negative log-likelihood')
#plt.title('')
plt.xlim([1, 5000])
plt.legend()
plt.show()












