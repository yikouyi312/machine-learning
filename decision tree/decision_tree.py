class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''

    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None


# import necessary package
import sys
import numpy as np


# read data
def read_data(file):
    return np.genfromtxt(file, delimiter="\t", dtype=str, encoding=None)


# calculate entropy, data: last co
def cal_entropy(data):
    # find the number of label result
    label = np.sort(np.unique(data))
    entropy = 0
    for i in label:
        entropy = entropy - np.mean(data == i) * np.log2(np.mean(data == i))
    return entropy


# calculate mutual information, I(data; X)
def cal_mutual_info(data, X):
    entropy = cal_entropy(data[:, -1])
    Info = entropy
    # find attribute index number
    for i in np.unique(data[:, X]):
        Info = Info - np.mean(data[:, X] == i) * cal_entropy(data[(data[:, X] == i), -1])
    return Info


# majority vote classifier
def majority_vote(data):
    # read the label of data
    data = data[:, -1]
    # find the number of label result
    result = np.sort(np.unique(data))
    max_index = ''
    max_count = 0
    # find the most common label
    for i in result:
        count = np.count_nonzero(data == i)
        if count >= max_count:
            max_count = count
            max_index = i
    return max_index  # return most common label


# train decision tree
def train_tree(data, max_depth):
    attribute = np.array(range(np.size(data[0, 0:-1])))
    root = train_recursive(data[1:, :], attribute, 0, max_depth, np.sort(np.unique(data[1:, -1])))
    return root


def train_recursive(data, attribute, layer, max_depth, result):
    p: Node = Node()
    if layer < max_depth:
        # Base case, leaf node
        # D is empty
        if np.size(data) == 0:
            p.vote = result[-1]
            return p
        else:
            if np.size(attribute) == 0:
                p.vote = majority_vote(data)
                return p
        # labels in D are identical
        if np.size(np.unique(data[:, -1])) == 1:
            p.vote = np.unique(data[:, -1])
            return p
        # for each attribute, all value are identical
        if np.shape(np.unique(data[:, attribute], axis=0))[0] == 1:
            p.vote = majority_vote(data)
            return p
        # Recursive step, internal node# find max splitting criterion, here is mutual information# split only when I>0
        maxInfo = 0
        for i in attribute:
            Info = cal_mutual_info(data, i)
            if Info > maxInfo:
                maxInfo = Info
                bestAttribute = i
        if maxInfo <= 0:
            p.vote = majority_vote(data)
            return p
        else:
            p.attr = bestAttribute
            # find attribute index number
            new_attribute = (attribute[attribute != bestAttribute])
            count = 0
            for i in np.sort(np.unique(data[:, bestAttribute])):
                new_data = data[(data[:, bestAttribute] == i), :]
                if count == 0:
                    # print([layer, bestAttribute])
                    p.left = train_recursive(new_data, new_attribute, layer + 1, max_depth, result)
                    count = count + 1
                else:
                    # print([layer, bestAttribute])
                    p.right = train_recursive(new_data, new_attribute, layer + 1, max_depth, result)
            return p
    else:
        p.vote = majority_vote(data)
        return p


# predict
def predict_tree(data, root):
    attribute = np.array(range(np.size(data[0, 0:-1])))
    temp = np.reshape(data[:, -1], (np.size(data[:, -1]), 1))
    data = np.concatenate((data, temp), axis=1)
    index = (data[1:, -1] != None)
    predict = predict_recursive(data[1:, :], attribute, root, index)[:, -1]
    return predict


def predict_recursive(data, attribute, node, index):
    # case 2 leaf node
    if (node.left is None) and (node.right is None):
        data[index, -1] = node.vote
        return data
    else:
        # case 1 internal node
        if node.attr in attribute:
            count = 0
            for i in np.sort(np.unique(data[:, node.attr])):
                new_attribute = (attribute[attribute != node.attr])
                if count == 0:
                    new_node = node.left
                else:
                    new_node = node.right
                new_index = np.multiply((data[:, node.attr] == i), index)
                data = predict_recursive(data, new_attribute, new_node, new_index)
                count = count + 1
        return data


def print_tree(data, tree):
    head = data[0, :]
    data = data[1:, :]
    label = np.sort(np.unique(data[1:, -1]))
    attribute = []
    for i in range(np.size(head) - 1):
        attribute = np.concatenate((attribute, np.sort(np.unique(data[:, i]))))
    attribute = np.reshape(attribute, (np.size(head) - 1, 2))
    print('[%d %s / %d %s]' % (np.count_nonzero(data[:, -1] == label[0]),
                               label[0], np.count_nonzero(data[:, -1] == label[1]), label[1]))
    layer = 1
    traversal(tree, data, head, label, layer, attribute)
    return


def traversal(root, data, head, label, layer, attribute):
    if root.vote is None:
        # First print the data of node
        count = 0
        for j in attribute[root.attr]:
            for i in range(layer):
                print('|', end=' ')
            print('%s = ' % (head[root.attr]), end=' ')
            print('%s: ' % j, end=' ')
            new_data = data[(data[:, root.attr] == j), :]
            print('[%d %s / %d %s]' % (np.count_nonzero(new_data[:, -1] == label[0]),
                                       label[0], np.count_nonzero(new_data[:, -1] == label[1]), label[1]))
            if count == 0:
                traversal(root.left, new_data, head, label, layer + 1, attribute)
                count = count + 1
            else:
                traversal(root.right, new_data, head, label, layer + 1, attribute)
        return
    else:
        return


def decision_tree(train_input, test_input, max_depth: int, train_out, test_out, metrics_out):
    # read data
    train_data = read_data(train_input)
    # train decision tree
    tree = train_tree(train_data, max_depth)
    # predict on train data
    predict_train = predict_tree(train_data, tree)
    # output in train_out
    np.savetxt(train_out, predict_train, fmt="%s")
    # read test data
    test_data = read_data(test_input)
    # predict on test data
    predict_test = predict_tree(test_data, tree)
    # output in test_out
    np.savetxt(test_out, predict_test, fmt="%s")
    # calculate the error rate
    train_error_rate = np.mean(predict_train != train_data[1:, -1])
    test_error_rate = np.mean(predict_test != test_data[1:, -1])
    # output the error rate in metrics_out
    np.savetxt(metrics_out, np.array([['error(train): %f' % train_error_rate], ['error(test): %f' % test_error_rate]]),
               fmt='%s')
    #print(train_error_rate, test_error_rate)
    print_tree(train_data, tree)
    return


# train_input = 'education_train.tsv'
# test_input = 'education_test.tsv'
# train_out = 'train.labels'
# test_out = 'test.labels'
# metrics_out = 'metrics.txt'
# decision_tree(train_input, test_input, 6, train_out, test_out, metrics_out)

if __name__ == '__main__':
    # arguments
    train_file: str = sys.argv[1]
    test_file: str = sys.argv[2]
    max_depth: int = int(sys.argv[3])
    train_out: str = sys.argv[4]
    test_out: str = sys.argv[5]
    metrics_out: str = sys.argv[6]
    decision_tree(train_file, test_file, max_depth, train_out, test_out, metrics_out)
