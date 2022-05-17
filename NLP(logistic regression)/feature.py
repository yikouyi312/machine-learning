import csv
import sys

import numpy as np
from numpy import ndarray

VECTOR_LEN = 300  # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt


################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An N x 2 np.ndarray. N is the number of data points in the tsv file. The
        first column contains the label integer (0 or 1), and the second column
        contains the movie review string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_dictionary(file):
    """
    Creates a python dict from the model 1 dictionary.

    Parameters:
        file (str): File path to the dictionary for model 1.

    Returns:
        A dictionary indexed by strings, returning an integer index.
    """
    dict_map = np.loadtxt(file, comments=None, encoding='utf-8',
                          dtype=f'U{MAX_WORD_LEN},l')
    return {word: index for word, index in dict_map}


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file for model 2.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map

#save data for model1
def save_tsv_dataset(data, file):
    csvfile = open(file, 'w', newline='\n')
    tsv_output = csv.writer(csvfile, delimiter='\t')
    tsv_output.writerows(data)
    csvfile.close()

#save data for model2
def save_tsv_dataset2(data, file):
    csvfile = open(file, 'w', newline='\n')
    tsv_output = csv.writer(csvfile, delimiter='\t')
    tsv_output.writerows("{:.6f}".format(data.str()))
    csvfile.close()

#Model 1 : bag_of_words
def bagofwords(data, dict_value):
    n = len(dict_value)
    m = len(data)
    format_data = np.intc(np.zeros((m, n+1)))
    for i in range(m):
        format_data[i][0] = data[i][0]
        sentence = set(data[i][1].split())
        for words in sentence:
            if words in dict_value:
                format_data[i][dict_value[words]+1] = 1
    return format_data

#Model2: Word Embeddings
def wordembeddings(data, dict_value):
    m = len(data)
    format_data: ndarray = np.zeros((m, VECTOR_LEN+1))
    for i in range(m):
        format_data[i][0] = data[i][0]
        sentence = data[i][1].split()
        J = 0
        for words in sentence:
            if words in dict_value:
                J += 1
                format_data[i][1:] += dict_value[words]
        format_data[i][1:] = 1 / J * format_data[i][1:]
    return np.round(format_data, 6)


train_input = 'smalldata/test_data.tsv'
validation_input = 'smalldata/valid_data.tsv'
test_input = 'smalldata/test_data.tsv'
dict_input = 'dict.txt'
feature_dictionary_input = 'word2vec.txt'
train_output = 'formattedtrain.tsv'
validation_output = 'formattedvalid.tsv'
test_output = 'formattedtest.tsv'
feature_flag = 2
# def convert(train_input: object, validation_input: object, test_input: object, dict_input: object,
#             feature_dictionary_input: object,formatted_train_out: object,
#             formatted_validation_out: object, formatted_test_out: object, feature_flag: object) -> object:
#     # read data
train = load_tsv_dataset(train_input)
valid = load_tsv_dataset(validation_input)
test = load_tsv_dataset(test_input)

# Model 1, Bag-of-words
if feature_flag == 1:
    dict_value = load_dictionary(dict_input)
    #format train data
    format_data = bagofwords(train, dict_value)
    save_tsv_dataset(format_data, formatted_train_out)
    #format validation data
    format_data = bagofwords(valid, dict_value)
    save_tsv_dataset(format_data, formatted_validation_out)
    #format test data
    format_data = bagofwords(test, dict_value)
    save_tsv_dataset(format_data, formatted_test_out)
else:
    #Model 2: Word Embeddings
    dict_value = load_feature_dictionary(feature_dictionary_input)
    #format train data
    format_data = wordembeddings(train, dict_value)
    save_tsv_dataset(format_data, formatted_train_out)
    # format validation data
    format_data = wordembeddings(valid, dict_value)
    save_tsv_dataset(format_data, formatted_validation_out)
    # format test data
    format_data = wordembeddings(test, dict_value)
    save_tsv_dataset(format_data, formatted_test_out)

# if __name__ == "__main__":
#     # arguments
#     train_input: str = sys.argv[1]
#     validation_input: str = sys.argv[2]
#     test_input: str = sys.argv[3]
#     dict_input: str = sys.argv[4]
#     feature_dictionary_input: str = sys.argv[5]
#     formatted_train_out: str = sys.argv[6]
#     formatted_validation_out: str = sys.argv[7]
#     formatted_test_out: str = sys.argv[8]
#     feature_flag: int = int(sys.argv[9])
#     convert(train_input, validation_input, test_input, dict_input, feature_dictionary_input,
#             formatted_train_out, formatted_validation_out, formatted_test_out, feature_flag)