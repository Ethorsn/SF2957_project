# For intro to AI gym, see https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym

# For blackjack and reinforcement learning, see Example 5.1 p. 76 in course book
# In this example, an infinite deck is used

import blackjack_extended as bjk
import blackjack_base as bjk_base
from math import inf
import Q_learning as ql
import sys
import os

import plotting as pl
from mc_prediction import mc_prediction
import time

if __name__ == "__main__":
    # matplotlib.style.use('ggplot')
    decks = inf
    directory = "{}/data".format(sys.path[0])

    if not os.path.exists(directory):
        os.makedirs(directory)
    path_fun = lambda x: "{}/{}_{}.txt".format(directory,x, decks)
    # set seed
    seed = 31233
    # init envs.
    env = bjk.BlackjackEnvExtend(decks = decks, seed=seed)
    sum_env = bjk_base.BlackjackEnvBase(decks = decks, seed=seed)

    # init constants
    alpha = 0.618
    n_sims = 10000000
    printall = False
    epsilon = 0.5
    init_val = 0.0
    # Q-learning with "correct" state representation
    start_time_expanded = time.time()
    Q, avg_reward, state_count = ql.learn_Q(
        env, n_sims, alpha, epsilon = epsilon, init_val = init_val,
        episode_file=path_fun("hand_state"))
    print("Number of explored states: " + str(len(Q)))
    print("Cumulative avg. reward = " + str(avg_reward))
    time_to_completion_expanded = time.time() - start_time_expanded

    print("----- Starting training for sum-based state space -----")
    # Q-learning with player sum state representation
    start_time_sum = time.time()
    sumQ, sum_avg_reward, sum_state_count = ql.learn_Q(
        sum_env, n_sims, alpha, epsilon = epsilon, init_val = init_val,
        episode_file=path_fun("sum_state"))
    time_to_completion_sum = time.time() - start_time_sum
    print("Number of explored states (sum states): " + str(len(sumQ)))
    print("Cumulative avg. reward = " + str(sum_avg_reward))
    print("Training time: \n " +
          "Expanded state space: {} \n Sum state space: {}".format(
              time_to_completion_expanded, time_to_completion_sum))
    """
    n_mcsim = 10000
    VQ = mc_prediction(lambda x: ql.Q_policy(x, Q, env), env, n_mcsim)
    VQ_conv = ql.convert_to_sum_states(VQ, True, 0)
    VQ_conv_filt = {state: VQ_conv[state] for state in V10k_sum if state[0] > 11}
    pl.plot_value_function(VQ_conv_filt, title="10,000 Steps")
    """
