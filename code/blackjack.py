import gym.envs.toy_text.blackjack as bj
import gym.spaces as spaces
import numpy as np

deck_values = [x for x in range(11)]

def sum_player_hand():
    return np.dot(deck, deck_values)

def draw_player_hand():
    hand = np.zeros(len(deck_values), int)
    hand[[bj.draw_card, bj.draw_card]] += 1
    return hand

def is_player_bust(hand):
    return sum_player_hand(hand) > 21

def player_score(hand):
    return 0 if is_player_bust(hand) else sum_player_hand(hand)

# Define the following functions for consistency
def sum_dealer_hand(hand):
    return bj.sum_hand(hand)

def dealer_score(hand)
    return bj.score(hand)

def draw_dealer_hand(n):
    return bj.draw_hand(n)

class BlackjackEnvExtend(bj.BlackjackEnv):
    """
    Class which extends OpenAI BlackJackEnv class such that it is a proper
    Markov decision process.

    Observation space is expanded, the player now sees the number of cards
    it is holding of each type.
    """
    def __init__(self, natural=True):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
                spaces.MultiDiscrete(10),
                spaces.Discrete(11)))
        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player[bj.draw_card(self.np_random)] += 1
            if is_player_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_dealer_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(player_score(self.player), dealer_score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (self.player, self.dealer[0])

    def reset(self):
        self.dealer = draw_dealer_hand(self.np_random)
        self.player = draw_player_hand(self.np_random)
        return self._get_obs()

