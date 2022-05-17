############ Welcome to HW7 ############
# TODO: Andrew-id: shuxianx@andrew.cmu.edu


# Imports
# Don't import any other library
import numpy as np
from utils import make_dict, parse_file, get_matrices, write_predictions, write_metrics
import argparse
import logging

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

#LogSumExpTrick
def log_sum_exp(para):
    m = np.max(para, axis=1)
    m1 = m.reshape((np.size(m), 1))
    return m + np.log(np.sum(np.exp(para-m1), axis=1))
def log_sum_exp1(para):
    m = np.max(para)
    return m + np.log(np.sum(np.exp(para-m)))

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
        for j in range(len(sentences[i])-1, -1, -1):
            if j == len(sentences[i]) - 1:
                #lnbeta[:, j] = [0]*len(keys)
                continue
            else:
                lnbeta[:, j] = log_sum_exp(lnbeta[:, j+1] + np.log(transition) + np.log(emission[:, word_dict[sentences[i][j+1]]]))

        # 4. Calculate probabilities and predictions
        #Computiong the Log Likelihood of a Sequence
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
    return

if __name__ == "__main__":
    args = parser.parse_args()
    # args = parser.parse_args(['en_data/validation.txt', 'en_data/index_to_word.txt', 'en_data/index_to_tag.txt',
    #                          'en_data/hmminit1.txt', 'en_data/hmmemit1.txt', 'en_data/hmmtrans1.txt',
    #                            'en_data/predicted1.txt', 'en_data/metrics1.txt'])
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)
