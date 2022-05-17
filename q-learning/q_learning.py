import argparse
import numpy as np
from environment import MountainCar, GridWorld


# NOTE: We highly recommend you to write functions for...
# - converting the state to a numpy array given a sparse representation
def convert(env, state, environment):
    sparse_rep = [0]*env.state_space
    if environment != 'mc':
        for i, j in state.items():
            sparse_rep[i] = 1
    else:
        if env.mode == 'raw':
            for i, j in state.items():
                sparse_rep[i] = j
        else:
            for i, j in state.items():
                sparse_rep[i] = 1
    return sparse_rep

# - determining which state to visit next
# - determining which action should do
def epsilon_greedy(R, epsilon, env):
    if np.random.random() > 1-epsilon:
        action = np.random.randint(0, len(R))
    else:
        action = np.argmax(R)
    return action

def main(args):
    # Command line inputs
    mode = args.mode
    weight_out = args.weight_out
    returns_out = args.returns_out
    episodes = args.episodes
    max_iterations = args.max_iterations
    epsilon = args.epsilon
    gamma = args.gamma
    learning_rate = args.learning_rate
    debug = args.debug

    # We will initialize the environment for you:
    if args.environment == 'mc':
        env = MountainCar(mode=mode, debug=debug)
    else:
        env = GridWorld(mode=mode, debug=debug)

    # TODO: Initialize your weights/bias here
    weights = np.zeros((env.state_space, env.action_space))  # Our shape is |A| x |S|, if this helps.
    bias = 0
    # If you decide to fold in the bias (hint: don't), recall how the bias is
    # defined!
    total_reward = []
    returns = []  # This is where you will save the return after each episode
    for episode in range(episodes):
        # Reset the environment at the start of each episode
        next_state = env.reset() # `state` now is the initial state
        to_reward = 0
        for it in range(max_iterations):
            # TODO: Fill in what we have to do every iteration
            # Hint 1: `env.step(ACTION)` makes the agent take an action
            #         corresponding to `ACTION` (MUST be an INTEGER)
            # Hint 2: The size of the action space is `env.action_space`, and
            #         the size of the state space is `env.state_space`
            # Hint 3: `ACTION` should be one of 0, 1, ..., env.action_space - 1
            # Hint 4: For Grid World, the action mapping is
            #         {"up": 0, "down": 1, "left": 2, "right": 3}
            #         Remember when you call `env.step()` you have to pass
            #         the INTEGER representing each action!
            sparse_state = convert(env, next_state, args.environment)
            q_saw = [np.matmul(sparse_state, weights[:, action]) + bias for action in range(env.action_space)]
            cur_action = epsilon_greedy(q_saw, epsilon, env)
            next_state, reward, done = env.step(cur_action)
            to_reward += reward
            next_sparse_state = convert(env, next_state, args.environment)
            next_q = [np.matmul(next_sparse_state, weights[:, action]) + bias for action in range(env.action_space)]
            weights[:, cur_action] = weights[:, cur_action] - learning_rate*(q_saw[cur_action] - (reward + gamma * np.max(next_q))) * np.array(sparse_state)
            bias = bias - learning_rate*(q_saw[cur_action] - (reward + gamma * np.max(next_q)))
            if done:
                break
        total_reward.append(to_reward)
    # TODO: Save output files
    #save data
    with open(weight_out, 'w') as f:
        f.write('{}\n'.format(bias))
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                f.write('{}\n'.format(weights[i][j]))
    np.savetxt(returns_out, total_reward, fmt="%s")
    return

if __name__ == "__main__":
    # No need to change anything here
    parser = argparse.ArgumentParser()
    parser.add_argument('environment', type=str, choices=['mc', 'gw'],
                        help='the environment to use')
    parser.add_argument('mode', type=str, choices=['raw', 'tile'],
                        help='mode to run the environment in')
    parser.add_argument('weight_out', type=str,
                        help='path to output the weights of the linear model')
    parser.add_argument('returns_out', type=str,
                        help='path to output the returns of the agent')
    parser.add_argument('episodes', type=int,
                        help='the number of episodes to train the agent for')
    parser.add_argument('max_iterations', type=int,
                        help='the maximum of the length of an episode')
    parser.add_argument('epsilon', type=float,
                        help='the value of epsilon for epsilon-greedy')
    parser.add_argument('gamma', type=float,
                        help='the discount factor gamma')
    parser.add_argument('learning_rate', type=float,
                        help='the learning rate alpha')
    parser.add_argument('--debug', type=bool, default=False,
                        help='set to True to show logging')
    # main(parser.parse_args(['mc', 'raw', 'mc_raw_weight.out', 'mc_raw_returns.out',
    #                         '4', '200', '0.05', '0.99', '0.01']))
    # main(parser.parse_args(['gw', 'tile', 'gw_simple_weight.out', 'gw_simple_returns.out',
    #                         '1', '1', '0.0', '1', '1']))
    # main(parser.parse_args(['mc', 'tile', 'mc_tile_weight.out', 'mc_tile_returns.out',
    #                         '25', '200', '0.0', '0.99', '0.005']))
    main(parser.parse_args())
