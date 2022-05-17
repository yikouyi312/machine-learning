# calculate the label entropy at the root
import sys
import numpy as np


# read data
def read_data(file):
    return np.genfromtxt(file, delimiter="\t", dtype=str, encoding=None)


def inspection(input, output):
    data = read_data(input)
    # read the label of input data
    data = data[1:, -1]
    # find the number of label result
    label = np.sort(np.unique(data))
    entropy = 0
    max_label = {}
    max_count = 0
    for i in label:
        count = np.count_nonzero(data == i)
        entropy = entropy - np.mean(data == i) * np.log2(np.mean(data == i))
        if count >= max_count:
            max_count = count
            max_label = i
    error_rate = np.mean(data != max_label)
    np.savetxt(output, np.array([['entropy: %f' % entropy], ['error: %f' % error_rate]]), fmt='%s')


if __name__ == "__main__":
    # arguments
    input: str = sys.argv[1]
    output: str = sys.argv[2]
    inspection(input, output)
