#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import blackjack as bjk
import numpy as np
import pandas as pd
from collections import defaultdict
import random


def learn_Q(env, n_sims, gamma = 1, omega = 0.77, epsilon = 0.05,
            init_val = 0.0, Q_init = None, episode_file = None):
    """
    gamma: discount factor
    omega: polynomial learning rate parameter (Even-Dar & Mansour, 2003)
    epsilon: exploration probability parameter
    init_val: initiate Q-values to something other than 0?
    Q_init: pre-trained Q dict
    episode_file: save ave
    """

    # Can start with a previously trained Q dict
    if Q_init is None:
        Q = defaultdict(lambda: np.zeros(env.action_space.n) + init_val)
    else:
        Q = Q_init

    state_action_count = defaultdict(lambda: np.zeros(env.action_space.n,
                                                      dtype = int))
    avg_reward = 0.0

    # if we want to save the episode reward to a file,
    if episode_file:
        f = open(episode_file, "w+")
        f.write("episode,avg_reward\n")
    else:
        f = None
    for episode in range(1,n_sims + 1):
        done = False
        action_reward = 0.0
        episode_reward = 0.0
        state = env.reset()
        while not done:
            explore = random.random() < (epsilon / (1 + state_action_count[state].sum()))
            if state not in Q or explore:
                # Take a random action
                action = env.action_space.sample()
            else:
                # Take the best possible action
                action = np.argmax(Q[state])

            # Update the state-action count
            state_action_count[state][action] += 1

            # Draw the next state and reward of previous action
            state2, action_reward, done, info = env.step(action)

            # Update Q, state and episode reward
            alpha = 1 / state_action_count[state][action] ** omega # Even-Dar & Mansour (2003)
            Q[state][action] += alpha * (action_reward + np.max(Q[state2]) -
                                         Q[state][action])
            state = state2
            episode_reward += action_reward

        if episode % (n_sims // 100) == 0:
            print('Avg. reward after {} episodes: {}'.format(episode, avg_reward))
            if f:
                # append to the file which we want to save to
                f.write("{},{}\n".format(episode, str(avg_reward)))

        # Game is over
        avg_reward += (episode_reward - avg_reward) / (episode + 1)

    return Q, avg_reward, state_action_count


def Q_policy(state, Q, env):
    if state in Q:
        return np.argmax(Q[state])
    return env.action_space.sample()


def convert_to_sum_states(Q, fill_missing = True,
                          default_value = np.array([0, 0])):
    """
    Function which convert the expanded state spce to a sum-based state space
    """
    S = dict()
    n = defaultdict(int)
    for state, action_values in Q.items():
        sum_state = (bjk.sum_player_hand(state[0]),
                     state[1],
                     bjk.usable_ace(state[0]))
        if sum_state[0] > 21:
            continue
        if sum_state in S:
            S[sum_state] = (action_values + n[sum_state]  * S[sum_state]) / (n[sum_state] + 1)
            n[sum_state] += 1
        else:
            S[sum_state] = action_values
    if fill_missing:
        for player_sum in range(1, 22):
            for dealer_sum in range(1, 11):
                state0 = (player_sum, dealer_sum, False)
                state1 = (player_sum, dealer_sum, True)
                if state0 not in S:
                    S[state0] = default_value
                if state1 not in S:
                    S[state1] = default_value
    return S


"""
def print_Q(Q):
    for key, value in sorted(Q.items(), key = lambda x: (x[0][0], (x[0][1]))):
        if bjk.sum_player_hand(key[0]) <= 21:
            print('(my hand = {}, dealereupcard = {}) -> ' +
                  '(stick = {}, hit = {})'.format(key[0], key[1],
                                                  round(value[0], 2),
                                                  round(value[1], 2)))
"""
