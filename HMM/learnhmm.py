############ Welcome to HW7 ############
# TODO: Andrew-id: shuxianx@andrew.cmu.edu


# Imports
# Don't import any other library
import argparse
import collections

import numpy as np
from utils import make_dict, parse_file
import logging

# Setting up the argument parser
# don't change anything here

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to store the hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to store the hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to store the hmm_transition.txt (B) file')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


# Hint: You might find it useful to define functions that do the following:
# 1. Calculate the init matrix
# 2. Calculate the emission matrix
# 3. Calculate the transition matrix
# 4. Normalize the matrices appropriately

# TODO: Complete the main function
def main(args):
    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)

    # Parse the train file
    # Suggestion: Take a minute to look at the training file,
    # it always hels to know your data :)
    sentences, tags = parse_file(args.train_input)

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
    emission = np.zeros((len(keys), len(keys_x)))   # initial emission matrix
    for i in range(len(tags)):
        for j in range(len(tags[i])):
            emission[tag_dict[tags[i][j]], word_dict[sentences[i][j]]] += 1     # count the number
    emission = (emission + 1) / np.transpose(np.sum(emission + 1, axis=1)).reshape(len(keys), 1)
    # transition = # TODO: Construct your transition matrix
    transition = np.zeros((len(keys), len(keys)))   # initial transition
    for i in range(len(tags)):
        for j in range(1, len(tags[i])):
            transition[tag_dict[tags[i][j-1]], tag_dict[tags[i][j]]] += 1   # count the number
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


# No need to change anything beyond this point
if __name__ == "__main__":
    args = parser.parse_args()
    # args = parser.parse_args(['en_data/train.txt', 'en_data/index_to_word.txt', 'en_data/index_to_tag.txt',
    #                          'en_data/hmminit1.txt', 'en_data/hmmemit1.txt', 'en_data/hmmtrans1.txt'])
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)
