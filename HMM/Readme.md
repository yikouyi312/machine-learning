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
  - Transition Matrix, $B$
    - $B_{jk} = P(Y_t = s_k | Y_{t-1} = s_j)$
  - Emission Matrix, $A$
    - $A_{jk} = P(X_t = k | Y_t = s_j)$
  - Initialization Matrix for $Y_1$, $\pi$     
    - $\pi_j = P(Y_1 = s_j)$
  - Maximum product of all the probabilities taken through path $Y_1, \cdots, Y_{t-1}$ that $Y_t = s_k$, $w_t(s_k)$
    - $w_t(s_k) = max_{y_1, \cdot, y_{t-1}} P(x_{1:t}, y_{1:t-1}, Y_t = s_k)$ 
  - Backpointer that stores the path through hidden states that gives us the highest product, $b_t(s_k)$
    - $b_t(s_k) = argmax_{y_1, \cdot, y_{t-1}} P(x_{1:t}, y_{1:t-1}, Y_t = s_k)$
* $P(Y_t = s_j|x_{1:T})\propto P(Y_t = s_j, x_{t+1:T}|x_{1:t})\propto P(Y_t = s_j|x_{1:t})P(x_{t+1:T}| Y_t = s_j, x_{1:t})\propto P(Y_t = s_j|x_{1:t})P(x_{t+1:T}| Y_t = s_j)\propto P(Y_t = s_j, x_{1:t})P(x_{t+1:T}| Y_t = s_j)  $
  - Forward Algorithm
    -  $\alpha_t(s_j) = P(Y_t = s_j, x_{1:t})$
    -  $\alpha_1(s_j) = \pi_j A_{j,x_1}$, $\alpha_t(s_j) = A_{j x_t}\sum_k B_{kj}\alpha_{t-1}(k)$
  - Backward Algorithm
    - $\beta_t(s_j) = P(x_{t+1:T}|Y_t = s_j)$
    - $\beta_T(s_j) = 1$, $\beta_t(s_j) = \sum_k A_{k, x_{t+1}} \beta_{t+1}(s_k) B_{jk}$ 

  - **Command Line Arguments** `$python feature.py <train input> <validation input> <test input> <dict input> <feature dictionary input> <formatted train out> <formatted validation out> <formatted test out> <feature flag>`


# Logistic regression classifier

* `lr.py`
  - **Command Line Arguments** `$python lr.py <formatted train input> <formatted validation input> <formatted test input> <train out> <test out> <metricsout> <numepoch> <learningrate>`
