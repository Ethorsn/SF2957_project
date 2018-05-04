# For intro to AI gym, see https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym

# For blackjack and reinforcement learning, see Example 5.1 p. 76 in course book
# In this example, an infinite deck is used

import gym
import blackjack as bjk
import numpy as np
from math import inf
import plotting as pl
from collections import defaultdict
from mc_prediction import mc_prediction
import random
import matplotlib

import Q_learning as ql








if __name__ == "__main__":

    matplotlib.style.use('ggplot')
    #env = gym.make('Blackjack-v0')
    decks = inf


    env = bjk.BlackjackEnvExtend(decks = decks)
    sum_env = gym.make('Blackjack-v0')


    alpha = 0.618
    n_sims = 100000
    printall = False

    # Q-learning with "correct" state representation
    Q, avg_reward, state_count = ql.learn_Q(env,
                                            n_sims,
                                            alpha,
                                            epsilon = 1,
                                            init_val = 0.0)
    print("Number of explored states: " + str(len(Q)))
    print("Cumulative avg. reward = " + str(avg_reward))

    # Convert to sum-state representation for 3D plotting

    # Calculate value of Q-policy
    n_mcsim = 10000
    VQ = mc_prediction(lambda x: ql.Q_policy(x, Q, env), env, n_mcsim)
    VQ_conv = ql.convert_to_sum_states(VQ, True, 0)
    VQ_conv_filt = {state: VQ_conv[state] for state in V10k_sum if state[0] > 11}
    pl.plot_value_function(VQ_conv_filt, title="10,000 Steps")

    # Q-learning with player sum state representation
    sumQ, sum_avg_reward, sum_state_count = ql.learn_Q(sum_env,
                                                       n_sims,
                                                       alpha,
                                                       epsilon = 0.05,
                                                       init_val = 0.0)
    print("Number of explored states (sum states): " + str(len(sumQ)))
    print("Cumulative avg. reward (sum states)= " + str(sum_avg_reward))


