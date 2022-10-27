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

- **Inspecting the data**
  - **_inspection.py_** calculate the label entropy at the root(i.e. the entropy of the labels before any splits) and the error rate (the percent of incorrectly classified instance) of classifying using a majority vote (picking the label with the most examples). 
  - **Command Line Arguments** `$python inspection.py <input> <output>`

- **Decision Tree Learner**
  - **_decision_tree.py_** learn a decision tree with a specified maximum depth.
  - **requirements:**
    - Use mutual information to determine which attribute to split on. For a split on arrtibute X, `I(Y;X)= H(Y)-H(Y|X)=H(Y)-P(X=0)H(Y|X=0)-P(X=1)H(Y|X=1)`.
    - As a stopping rule, only split on an attribute if the mutual information is > 0. If different columns have equal values for mutual information, split on the **first** column to break **ties**.
    - Use a majority vote of the labels at each leaf to make classification decisions. If the vote is **tied**, choose the label that comes **last** in the lexicographical order.
  - **Command Line Arguments** `$python decision_tree.py <train input> <test input> <max depth> <train output> <test out> <metrics out>`
