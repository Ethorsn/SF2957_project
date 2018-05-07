import blackjack_base as bj
import gym.spaces as spaces
import numpy as np
from gym.utils import seeding
from math import inf

deck = np.array([1,2,3,4,5,6,7,8,9,10,10,10,10])
deck_values = np.array([x for x in range(1, 11)])


def sum_player_hand(hand):
    return np.dot(deck_values, hand) + 10 * usable_ace(hand)

def sum_with_ace(hand):
    hand_sum = np.dot(deck_values, hand)
    return hand_sum if not hand_sum + 10 <= 21 else hand_sum + 10

def usable_ace(hand):
    return hand[0] > 0 and np.dot(deck_values, hand) + 10 <= 21

def is_player_bust(hand):
    return sum_player_hand(hand) > 21

def player_score(hand):
    return 0 if is_player_bust(hand) else sum_player_hand(hand)

# Define the following functions for consistency
def sum_dealer_hand(hand):
    return bj.sum_hand(hand)

def dealer_score(hand):
    return bj.score(hand)

def is_natural(hand):
    # A hand is a natural blackjack if it has 2 cars which total 21
    return True if sum(hand) == 2 & sum_player_hand(hand) == 21 else False

class BlackjackEnvExtend(bj.BlackjackEnvBase):
    """
    Class which extends OpenAI BlackJackEnv class such that it is a proper
    stationary Markov decision process.

    Observation space is expanded, the agent now sees the number of cards
    it is holding at each state.
    """
    def __init__(self, decks = inf, seed=3232, natural=True):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
        # MultiDiscrete is a vector of the number of possible values per element
                spaces.MultiDiscrete([11,11,8,6,5,4,4,3,3,3]),
                spaces.Discrete(11)))
        self.seed(seed)
        # initialize the number of cards to have of each deck
        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        self.decks = decks # number of decks
        # Start the first game
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player[self.draw_card(self.np_random) - 1] += 1 # Subtract 1 due to 0-based indexing
            if is_player_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_dealer_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card(self.np_random))
            reward = bj.cmp(player_score(self.player), dealer_score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def draw_player_hand(self, np_random):
        hand = np.zeros(len(deck_values), int)
        hand[self.draw_card(np_random) - 1] += 1
        hand[self.draw_card(np_random) - 1] += 1
        return hand

    def draw_dealer_hand(self, n):
        return self.draw_card(n)

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self):
        return (tuple(self.player), self.dealer[0])

    def reset(self):
        self.construct_deck()
        self.dealer = [self.draw_dealer_hand(self.np_random)]
        self.player = self.draw_player_hand(self.np_random)
        return self._get_obs()

