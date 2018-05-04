#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import blackjack as bjk
import numpy as np
import pandas as pd
from collections import defaultdict
import random



def learn_Q(env, n_sims, alpha, init_val = 0.0, epsilon = 0.05, Q_init = None, episode_file=None):
    """
    
    """
    
    if Q_init is None:
        Q = defaultdict(lambda: np.zeros(env.action_space.n) + init_val)
    else:
        Q = Q_init
        
    state_count = defaultdict(int)
    avg_reward = 0.0
    eps_decay = 1.0
    
    # if we want to save the episode reward to a file, 
    if episode_file:
        f = open(episode_file, "w+")
        f.write("episode, episode_reward")
        
    for episode in range(1,n_sims + 1):
        if episode > (n_sims // 10):
            eps_decay = 1 / episode
        done = False
        action_reward = 0.0
        episode_reward = 0.0
        state = env.reset()
        state_count[state] += 1
        while not done:
            if state in Q and random.random() > epsilon * eps_decay:
                # Take the best possible action
                action = np.argmax(Q[state])
            else:
                # Take a random action
                action = env.action_space.sample()
            # Draw the next state and reward of previous action
            state2, action_reward, done, info = env.step(action)

            # Update Q, state and episode reward
            Q[state][action] += alpha * (action_reward + np.max(Q[state2]) - Q[state][action])
            state = state2
            state_count[state] += 1
            episode_reward += action_reward
            
        # append to the file which we want to save to 
        if f:
            f.write(episode, episode_reward)
            
        if episode % (n_sims // 100) == 0:
            print('Avg. reward after {} episodes: {}'.format(episode, avg_reward))
        
        
        # Game is over
        avg_reward += (episode_reward - avg_reward) / (episode + 1)

    return Q, avg_reward, state_count


def print_Q(Q):
    for key, value in sorted(Q.items(), key = lambda x: (x[0][0], (x[0][1]))):
        if bjk.sum_player_hand(key[0]) <= 21:
            print('(my hand = {}, dealer upcard = {}) -> (stick = {}, hit = {})'.format(
                key[0], key[1],round(value[0], 2), round(value[1], 2)))

def Q_policy(state, Q, env):
    if state in Q:
        return np.argmax(Q[state])
    return env.action_space.sample()

def convert_to_sum_states(Q,
                          fill_missing = True,
                          default_value = np.array([0, 0])):
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

