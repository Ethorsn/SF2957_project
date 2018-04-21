# For intro to AI gym, see https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym

# For blackjack and reinforcement learning, see Example 5.1 p. 76 in course book
# In this example, an infinite deck is used

import gym
env = gym.make('Blackjack-v0')
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
Q = dict()

G = 0

alpha = 0.618

cumulative_reward = 0.0

n_sims = 100000

for episode in range(1,n_sims + 1):
    done = False
    G, reward = 0,0
    state = env.reset()
    while not done:
        if state not in Q:
            Q[state] = np.zeros(env.action_space.n)
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state]) #1
        state2, reward, done, info = env.step(action) #2
        if state2 not in Q:
            Q[state2] = np.zeros(env.action_space.n)
        Q[state][action] += alpha * (reward + np.max(Q[state2]) - Q[state][action]) #3
        G += reward
        state = state2
    cumulative_reward += G
    if episode % n_sims // 10 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G))

print("Number of explored states: " + str(len(Q)))
print("cumulative_reward = " + str(cumulative_reward))
