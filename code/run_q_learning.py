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
import matplotlib

matplotlib.style.use('ggplot')

if __name__ == "__main__":
    directory = "{}/data".format(sys.path[0])
    if not os.path.exists(directory):
        os.makedirs(directory)
    path_fun = lambda x: "{}/{}_{}.txt".format(directory,x, decks)
    # init constants
    omega = 0.77
    n_sims = 10 ** 7
    epsilon = 0.5
    init_val = 0.0
    warmup = n_sims//10
    # Directory to save plots in
    plot_dir = "{}report/figures/".format(sys.path[0][:-4])

    for decks in [1,2,6,8,inf]:
        print("----- deck number equal to {} -----".format(decks))
        # set seed
        seed = 31233
        # init envs.
        env = bjk.BlackjackEnvExtend(decks=decks, seed=seed)
        sum_env = bjk_base.BlackjackEnvBase(decks=decks, seed=seed)

        # Q-learning with "correct" state representation
        start_time_expanded = time.time()
        Q, avg_reward, state_action_count = ql.learn_Q(
            env, n_sims, omega = omega, epsilon = epsilon, init_val = init_val,
            episode_file=path_fun("hand_state"), warmup=warmup)
        print("Number of explored states: " + str(len(Q)))
        print("Cumulative avg. reward = " + str(avg_reward))
        time_to_completion_expanded = time.time() - start_time_expanded

        print("----- Starting training for sum-based state space -----")
        # Q-learning with player sum state representation
        start_time_sum = time.time()
        sumQ, sum_avg_reward, sum_state_action_count = ql.learn_Q(
            sum_env, n_sims, omega = omega, epsilon = epsilon, init_val = init_val,
            episode_file=path_fun("sum_state"), warmup=warmup)
        time_to_completion_sum = time.time() - start_time_sum
        print("Number of explored states (sum states): " + str(len(sumQ)))
        print("Cumulative avg. reward = " + str(sum_avg_reward))

        print("Training time: \n " +
              "Expanded state space: {} \n Sum state space: {}".format(
                  time_to_completion_expanded, time_to_completion_sum))

        # Convert Q (extended state) to sum state representation and make 3D plots
        Q_conv = ql.convert_to_sum_states(Q, env)
        V_conv = ql.convert_to_value_function(Q_conv)
        V_conv_filt = ql.fill_missing_sum_states(ql.filter_states(V_conv))
        pl.plot_value_function(V_conv_filt,
                               title = "Expanded state, " + str(decks) + " decks",
                               directory = plot_dir,
                               file_name = "3D_exp_" + str(decks) + "_decks.png")

        # Likewise make 3D plots for sumQ
        V_sum = ql.convert_to_value_function(sumQ)
        V_sum_filt = ql.fill_missing_sum_states(ql.filter_states(V_sum))
        pl.plot_value_function(V_sum_filt,
                               title = "Sum state, " + str(decks) + " decks",
                               directory = plot_dir,
                               file_name = "3D_sum_" + str(decks) + "_decks.png")
        # create line plots
        env_types = ["hand", "sum"]
        fig, lgd = pl.plot_avg_reward_episode(directory, env_types, [str(decks)])
        fig.savefig("{}/avgReturnEp_ndeck{}.png".format(plot_dir, decks),
                                bbox_extra_artists=(lgd,), bbox_inches='tight')
        matplotlib.pyplot.close(fig)
