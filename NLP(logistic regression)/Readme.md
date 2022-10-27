# Task
* Implement a working Natural Language Processing (NLP) system using binary logistic regression.

# Data
* Movie Review Polarity dataset
* see detail ` http://www.cs.cornell.edu/people/pabo/movie-review-data/`
* 0 for negative reviews and 1 for positive one
* format `label\tword1 word2 word3 ... wordN\n`

# Feature Engineer
* bag-of-words (BoW) model
  - $\phi_1(x^i) = 1_{occur}(x^i, \text{Vocab})$ 
  - indicates which words in vocabulary Vocab occur at least once in the i-th movie review $x^i$
* word embeddings model
  - represent each word by “embedding” it into a low-dimensional vector space, which may carry more information about the semantic meaning of the word.
  - *word2vec embedding*
  - word2vec.txt contains the word2vec embeddings of 15k words
  - format `word\tfeature1\tfeature2\t...feature300\n.`
  - $\phi_2(x^i) = \frac{1}{J} \sum \textit{word2vec}(xtrim_j^i)$
* Implements bag-of-words and word embeddings to transform raw training examples to formatted training examples.
  - `feature.py`
  - **Command Line Arguments** `$python feature.py <train input> <validation input> <test input> <dict input> <feature dictionary input> <formatted train out> <formatted validation out> <formatted test out> <feature flag>`


# Logistic regression classifier
* Implements a logistic regression classifier that takes in formatted training data and produces a label (either 0 or 1) that corresponds to whether each movie review was negative or positive.
* Maxmize likelihood $=$ Minimize negative conditional log-likelihood
* `lr.py`
  - **Command Line Arguments** `$python lr.py <formatted train input> <formatted validation input> <formatted test input> <train out> <test out> <metricsout> <numepoch> <learningrate>`
