# For intro to AI gym, see https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym

# For blackjack and reinforcement learning, see Example 5.1 p. 76 in course book
# In this example, an infinite deck is used

import gym
import blackjack as bjk
#env = gym.make('Blackjack-v0')
env = bjk.BlackjackEnvExtend()
env.reset()



# Show number of possible actions
env.action_space.n # 0=stick, 1=hit

# Observation space is 32 x 11 x 2
# 32 = the player's current sum
# 11 = the dealer's one showing card
# 2 = the player holds a usable ace
env.observation_space

obs_space_n = 32 * 11 * 2

import numpy as np


def learn_Q(env, n_sims, alpha, init_val = 0.0):
    Q = dict()

    avg_reward = 0.0

    for episode in range(1,n_sims + 1):
        done = False
        action_reward = 0.0
        episode_reward = 0.0
        state = env.reset()
        while not done:
            if state not in Q:
                # Initialize Q and take an action uniformly at random
                Q[state] = np.zeros(env.action_space.n) + init_val
                action = env.action_space.sample()
            else:
                # Take the best possible action
                action = np.argmax(Q[state])
            # Draw the next state and reward of previous action
            state2, action_reward, done, info = env.step(action)
            # If we haven't seen the new state before, initialize it for Q
            if state2 not in Q:
                Q[state2] = np.zeros(env.action_space.n) + init_val

            # Update Q, state and episode reward
            Q[state][action] += alpha * (action_reward + np.max(Q[state2]) - Q[state][action])
            state = state2
            episode_reward += action_reward

            if episode % (n_sims // 100) == 0:
                print('Avg. reward after {} episodes: {}'.format(episode,avg_reward))

        # Game is over
        avg_reward += (episode_reward - avg_reward) / (episode + 1)

    return Q, avg_reward


if __name__ == "__main__":
    alpha = 0.618
    avg_reward = 0.0
    n_sims = 10000

    Q1, avg1 = learn_Q(env, n_sims, alpha, init_val = 0.0)

    print("Number of explored states: " + str(len(Q1)))
    print("Cumulative avg. reward = " + str(avg1))
    for key, value in sorted(Q1.items(), key = lambda x: (x[0][0], (x[0][1]))):
        print(value[0])
        print(value[1])
        if bjk.sum_player_hand(key[0]) <= 21:
            print('(my hand = {}, dealer sum = {}) -> (stick = {}, hit = {})'.format(
                key[0], key[1],round(value[0], 2), round(value[1], 2)))
