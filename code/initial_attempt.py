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


matplotlib.style.use('ggplot')
#env = gym.make('Blackjack-v0')
decks = inf
env = bjk.BlackjackEnvExtend(decks = decks)

sum_env = gym.make('Blackjack-v0')

# Show number of possible actions
print(env.action_space.n) # 0=stick, 1=hit

# Observation space is 32 x 11 x 2
# 32 = the player's current sum
# 11 = the dealer's one showing card
# 2 = the player holds a usable ace
env.observation_space

obs_space_n = 32 * 11 * 2

def learn_Q(env, n_sims, alpha,
            init_val = 0.0, epsilon = 0.05,
            Q_init = None):
    if Q_init is None:
        Q = defaultdict(lambda: np.zeros(env.action_space.n) + init_val)
    else:
        Q = Q_init
    state_count = defaultdict(int)

    avg_reward = 0.0

    for episode in range(1,n_sims + 1):
        done = False
        action_reward = 0.0
        episode_reward = 0.0
        state = env.reset()
        state_count[state] += 1
        while not done:
            if state in Q and random.random() > epsilon:
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

        if episode % (n_sims // 100) == 0:
            print('Avg. reward after {} episodes: {}'.format(episode, avg_reward))

        # Game is over
        avg_reward += (episode_reward - avg_reward) / (episode + 1)

    return Q, avg_reward, state_count


def print_Q(Q):
    for key, value in sorted(Q.items(),
                             key = lambda x: (x[0][0], (x[0][1]))):
        if bjk.sum_player_hand(key[0]) <= 21:
            print('(my hand = {}, dealer upcard = {}) -> (stick = {}, hit = {})'.format(
                key[0], key[1],round(value[0], 2), round(value[1], 2)))

def Q_policy(state, Q):
    return np.argmax(Q[state])

def convert_to_sum_states(Q):
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
    return S



if __name__ == "__main__":
    alpha = 0.618
    n_sims = 100000
    printall = False

    Q, avg_reward, state_count = learn_Q(env, n_sims, alpha, epsilon = 0.05, init_val = 0.0)
    print("Number of explored states: " + str(len(Q)))
    print("Cumulative avg. reward = " + str(avg_reward))
    #print("Cumulative avg. reward = " + str(avg_reward))

    sum_Q = convert_to_sum_states(Q)
    print("Number of explored sum states: " + str(len(sum_Q)))

    V_10k = mc_prediction(lambda x: Q_policy(x, Q), env, 10000)
    plotting.plot_value_function(convert_to_sum_states(V_10k),
                                 title="10,000 Steps")

    V_500k = mc_prediction(lambda x: Q_policy(x, Q), env, 500000)
    plotting.plot_value_function(convert_to_sum_states(V_500k), title="500,000 Steps")


    if printall:
        print_Q(Q)

