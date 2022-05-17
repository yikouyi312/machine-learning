# Imports
# Don't import any other library
import numpy as np
from utils import make_dict, parse_file, get_matrices, write_predictions, write_metrics
import argparse
import logging
import matplotlib.pyplot as plt
import collections
from matplotlib.ticker import FormatStrFormatter

parser1 = argparse.ArgumentParser()
parser1.add_argument('train_input', type=str,
                     help='path to training input .txt file')
parser1.add_argument('index_to_word', type=str,
                     help='path to index_to_word.txt file')
parser1.add_argument('index_to_tag', type=str,
                     help='path to index_to_tag.txt file')
parser1.add_argument('init', type=str,
                     help='path to store the hmm_init.txt (pi) file')
parser1.add_argument('emission', type=str,
                     help='path to store the hmm_emission.txt (A) file')
parser1.add_argument('transition', type=str,
                     help='path to store the hmm_transition.txt (B) file')
parser1.add_argument('--debug', type=bool, default=False,
                     help='set to True to show logging')


def main1(args, seq):
    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)

    # Parse the train file
    # Suggestion: Take a minute to look at the training file,
    # it always hels to know your data :)
    sentences, tags = parse_file(args.train_input)
    sentences = sentences[:seq]
    tags = tags[:seq]

    logging.debug(f"Num Sentences: {len(sentences)}")
    logging.debug(f"Num Tags: {len(tags)}")

    # Train your HMM
    # init = # TODO: Construct your init matrix
    keys = list(tag_dict.keys())  # key of tag
    init = np.zeros((len(keys), 1))  # initial init
    init_list = [tags[i][0] for i in range(len(tags))]
    count = collections.Counter(init_list)
    for i in range(len(keys)):
        init[i] = (count[keys[i]] + 1) / (len(tags) + len(keys))
    # emission = # TODO: Construct your emission matrix
    keys_x = list(word_dict.keys())  # key of word
    emission = np.zeros((len(keys), len(keys_x)))  # initial emission matrix
    for i in range(len(tags)):
        for j in range(len(tags[i])):
            emission[tag_dict[tags[i][j]], word_dict[sentences[i][j]]] += 1  # count the number
    emission = (emission + 1) / np.transpose(np.sum(emission + 1, axis=1)).reshape(len(keys), 1)
    # transition = # TODO: Construct your transition matrix
    transition = np.zeros((len(keys), len(keys)))  # initial transition
    for i in range(len(tags)):
        for j in range(1, len(tags[i])):
            transition[tag_dict[tags[i][j - 1]], tag_dict[tags[i][j]]] += 1  # count the number
    transition = (transition + 1) / np.transpose(np.sum(transition + 1, axis=1)).reshape(len(keys), 1)

    # Making sure we have the right shapes
    # logging.debug(f"init matrix shape: {init.shape}")
    # logging.debug(f"emission matrix shape: {emission.shape}")
    # logging.debug(f"transition matrix shape: {transition.shape}")

    ## Saving the files for inference
    ## We're doing this for you :)
    ## TODO: Just Uncomment the following lines when you're ready!

    np.savetxt(args.init, init)
    np.savetxt(args.emission, emission)
    np.savetxt(args.transition, transition)

    return


# Setting up the argument parser
# don't change anything here
parser = argparse.ArgumentParser()
parser.add_argument('validation_input', type=str,
                    help='path to validation input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to the learned hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to the learned hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to the learned hmm_transition.txt (B) file')
parser.add_argument('prediction_file', type=str,
                    help='path to store predictions')
parser.add_argument('metric_file', type=str,
                    help='path to store metrics')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


# Hint: You might find it helpful to define functions
# that do the following:
# 1. Calculate Alphas
# 2. Calculate Betas
# 3. Implement the LogSumExpTrick
# 4. Calculate probabilities and predictions

# LogSumExpTrick
def log_sum_exp(para):
    m = np.max(para, axis=1)
    m1 = m.reshape((np.size(m), 1))
    return m + np.log(np.sum(np.exp(para - m1), axis=1))


def log_sum_exp1(para):
    m = np.max(para)
    return m + np.log(np.sum(np.exp(para - m)))


# TODO: Complete the main function
def main(args):
    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)

    # Parse the validation file
    sentences, tags = parse_file(args.validation_input)

    ## Load your learned matrices
    ## Make sure you have them in the right orientation
    ## TODO:  Uncomment the following line when you're ready!
    init, emission, transition = get_matrices(args)
    # TODO: Conduct your inferences
    keys = list(tag_dict.keys())  # key of tag
    keys_x = list(word_dict.keys())  # key of word
    y_pred = []
    loglikelihood = []

    for i in range(len(sentences)):
        # 1. Calculate Alphas
        lnalpha = np.zeros((len(keys), len(sentences[i])))
        for j in range(len(sentences[i])):
            if j == 0:
                lnalpha[:, j] = np.log(init) + np.log(emission[:, word_dict[sentences[i][j]]])
            else:
                # logem = np.log(emission[:, word_dict[sentences[i][j]]])
                # lnalpha[:, j] = log_sum_exp(logem.reshape((np.size(logem), 1)) +
                #                             lnalpha[:, j-1] + np.log(np.transpose(transition)))
                lnalpha[:, j] = np.log(emission[:, word_dict[sentences[i][j]]]) + \
                                log_sum_exp(lnalpha[:, j - 1] + np.log(np.transpose(transition)))

        # 2. Calculate Betas
        lnbeta = np.zeros((len(keys), len(sentences[i])))
        for j in range(len(sentences[i]) - 1, -1, -1):
            if j == len(sentences[i]) - 1:
                # lnbeta[:, j] = [0]*len(keys)
                continue
            else:
                lnbeta[:, j] = log_sum_exp(
                    lnbeta[:, j + 1] + np.log(transition) + np.log(emission[:, word_dict[sentences[i][j + 1]]]))

        # 4. Calculate probabilities and predictions
        # Computiong the Log Likelihood of a Sequence
        loglikelihood.append(log_sum_exp1(lnalpha[:, -1]))
        # Minimun Bayes Risk Prediction
        lnprob = lnalpha + lnbeta
        yhat = np.argmax(lnprob, axis=0)
        y_pred.append([keys[k] for k in yhat])

    # TODO: Generate your probabilities and predictions
    # predicted_tags = #TODO: store your predicted tags here (in the right order)
    predicted_tags = y_pred
    # avg_log_likelihood = # TODO: store your calculated average log-likelihood here
    # We'll calculate this for you
    avg_log_likelihood = np.mean(loglikelihood)
    ## Writing results to the corresponding files.
    ## We're doing this for you :)
    ## TODO: Just Uncomment the following lines when you're ready!
    accuracy = write_predictions(args.prediction_file, sentences, predicted_tags, tags)
    write_metrics(args.metric_file, avg_log_likelihood, accuracy)
    return avg_log_likelihood


seq = [10, 100, 1000, 10000]
train = []
valid = []
for i in seq:
    args1 = parser1.parse_args(['en_data/train.txt', 'en_data/index_to_word.txt', 'en_data/index_to_tag.txt',
                              'en_data/hmminit1.txt', 'en_data/hmmemit1.txt', 'en_data/hmmtrans1.txt'])
    main1(args1, i)
    args = parser.parse_args(['en_data/train.txt', 'en_data/index_to_word.txt', 'en_data/index_to_tag.txt',
                              'en_data/hmminit1.txt', 'en_data/hmmemit1.txt', 'en_data/hmmtrans1.txt',
                              'en_data/predicted_train.txt', 'en_data/metrics_train.txt'])
    train.append(main(args))

    args = parser.parse_args(['en_data/validation.txt', 'en_data/index_to_word.txt', 'en_data/index_to_tag.txt',
                              'en_data/hmminit1.txt', 'en_data/hmmemit1.txt', 'en_data/hmmtrans1.txt',
                              'en_data/predicted_valid.txt', 'en_data/metrics_valid.txt'])
    valid.append(main(args))

# plot1 = plt.figure(1)
# blue_line = plt.plot(list(range(1, 14000 + 1)), train, color='blue', label='Train Data')
# red_line = plt.plot(list(range(1, 3300 + 1)), valid, color='red', label='Validation Data')
# plt.xlabel('number of sequences')
# plt.ylabel('average log likelihood')
# plt.title('Average Log Likelihood vs number of sequences')
# # plt.xlim([0, 200])
# plt.legend()
# plt.show()

plot1 = plt.figure(1)
blue_line = plt.plot([10, 100, 1000, 10000], train, color='blue', marker='o', label='Train Data')
red_line = plt.plot([10, 100, 1000, 10000], valid, color='red', marker='o', label='Validation Data')
plt.xlabel('number of sequences')
plt.ylabel('average log likelihood')
plt.title('Average Log Likelihood vs number of sequences')
# plt.xlim([0, 10000])
plt.legend()
plt.show()

