# Task
* Implement a neural network to classify images using a single hidden layer neural network

# Data
* Optical Character Recognition (OCR) dataset
* Original images are black-and-white (not grayscale), the pixel values are either 0 or 1

# Model
* Single-hidden-layer neural network with a sigmoid activation function for the hidden layer, and a softmax on the output layer.
* Loss Function
  - average cross entropy over the training dataset $D = {(x(i), y(i))}$
  - $J(\alpha, \beta) = -\frac{1}{N}\sum\sum y^i_k \log (\hat{y}_k^i)$
* Optimizer
  - SGD (stochastic gradient descent)
  - $J_{SGD}(\alpha, \beta) = -\sum y^i_k \log (\hat{y}_k^i)$

  - $\phi_1(x^i) = 1_{occur}(x^i, \text{Vocab})$ 
  - indicates which words in vocabulary Vocab occur at least once in the i-th movie review $x^i$
* Adagrad
  - Adagrad update
  - $s^i_{t+1} = s^i_t + \frac{\partial J(\theta_t)}{\partial \theta^i_t}\odot \frac{\partial J(\theta_t)}{\partial \theta^i_t}$, $\odot$ element-wise
  - $\theta^i_{t+1} = \theta^i_t - \frac{\eta}{\sqrt{s^i_{t+1}+\epsilon}}\odot \frac{\partial J(\theta_t)}{\partial \theta^i_t}$, $\eta$ learning-rate, $\epsilon$ parameter

# Code
* `neuralnet.py`
  - **Command Line Arguments** `$python3 neuralnet.py  <train input> validation input> <train out> <validation out> <metrics out> <num epoch> <hidden units> <init flag> <learning rate>`

