# Task
* Implement a named entity recognition system using Hidden Markov Models (HMMs)
* Named entity recognition (NER) is the task of classifying named entities, typically proper nouns, into pre-defined categories, such as person, location, or organization. 
# Data
* The WikiANN dataset provides labelled entity data for Wikipedia articles in 282 languages
* WikiANN is a “silver standard” dataset that was generated without human labelling. 
  The English Abstract Meaning Representation (AMR) corpus and DBpedia features were used to train an automatic classifier to label Wikipedia articles. 
  These labels were then propagated throughout other Wikipedia articles using the Wikipedia’s cross-language links and redirect links. 
* Afterwards, another tagger that self-trains on the existing tagged entities was used to label all other mentions of the same entities, 
  even those with different morphologies (prefixes and suffixes that modify a word in other languages). 
  Finally, the amassed training examples were filtered by “commonness” and “topical relatedness” to pick more relevant training data.
* format `<Word0>\t<Tag0>\n<Word1>\t<Tag1>\n ... <WordN>\t<TagN>\n`

# Model
* Hidden Markov Model
  - Evaluation and Decoding
  - Forward Backward Algorithm and Minimal Bayes Risk Decoding
* Notation
  - Observe state
    - $x_1, \cdots, x_T$
  - Hidden state
    - $Y_1, \cdots, Y_T$  
  - Transition Matrix 
    - $B$
    - $B_{jk} = P(Y_t = s_k | Y_{t-1} = s_j)$
  - Emission Matrix
    - $A$
    - $A$   
* $P(Y_t = s_j|x_{1:T})\propto P(Y_t = s_j, x_{t+1:T}|x_{1:t})\propto P(Y_t = s_j|x_{1:t})P(x_{t+1:T}| Y_t = s_j, x_{1:t})\propto P(Y_t = s_j|x_{1:t})P(x_{t+1:T}| Y_t = s_j)\propto P(Y_t = s_j, x_{1:t})P(x_{t+1:T}| Y_t = s_j)  $
  - Forward Algorithm
    -  $\alpha_t(s_j) = P(Y_t = s_j, x_{1:t})$
    -  
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
