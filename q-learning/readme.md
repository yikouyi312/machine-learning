# Task
* Implement Q-learning with linear function approximation to solve the mountain car environment
*  In Mountain Car need control a car that starts at the bottom of a valley. Our goal is to reach the flag at the top righ
# Setting
* The state of the environment is represented by two variables, position and velocity. 
  - position can be between [−1.2, 0.6] (inclusive).
  - velocity can be between [−0.07, 0.07] (inclusive). 
  - These are just measurements along the x-axis.
* The actions that you may take at any state are {0, 1, 2}, where each number corresponds to an action
  - (0)pushing the car left.
  - (1) doing nothing.
  - (2) pushing the car right.

# Model
* Q-learning with Linear Approximations

  - **Command Line Arguments** `$python feature.py <train input> <validation input> <test input> <dict input> <feature dictionary input> <formatted train out> <formatted validation out> <formatted test out> <feature flag>`


#  Q-learning algorithm
* `q_learning.py`
  - **Command Line Arguments** `$ python q_learning.py <env> <mode> <weight out> <returns out> <episodes> <max iterations> <epsilon> <gamma> <learning rate>`
