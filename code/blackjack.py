import gym.envs.toy_text.blackjack as bj
import gym.spaces as spaces
import numpy as np
from gym.utils import seeding

deck = np.array([1,2,3,4,5,6,7,8,9,10,10,10,10])
deck_values = np.array([x for x in range(1, 11)])
deck_values_ace = deck_values.copy()
deck_values_ace[0] = 11


def _sph(hand, deck_val):
    return np.dot(deck_val, hand)

def sum_player_hand(hand):
    return _sph(hand, deck_values) if (abs(_sph(hand, deck_values) - 21) <
                                       abs(_sph(hand, deck_values_ace) - 21)) else _sph(hand, deck_values_ace)

def is_player_bust(hand):
    return sum_player_hand(hand) > 21

def player_score(hand):
    # This function should be modified so that we can deal with aces
    return 0 if is_player_bust(hand) else sum_player_hand(hand)

# Define the following functions for consistency
def sum_dealer_hand(hand):
    return bj.sum_hand(hand)

def dealer_score(hand):
    return bj.score(hand)

def draw_dealer_hand(n):
    return bj.draw_hand(n)

def is_natural(hand):
    return True if sum(hand) == 2 & sum_player_hand(hand) == 21 else False

class BlackjackEnvExtend(bj.BlackjackEnv):
    """
    Class which extends OpenAI BlackJackEnv class such that it is a proper
    Markov decision process.

    Observation space is expanded, the player now sees the number of cards
    it is holding of each type.
    """
    def __init__(self, decks, seed=3232, natural=True):
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
        # Start the first game
        self.reset(decks)

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

    def construct_deck(self,decks):
        self.cards_in_deck = {x: decks for x in deck_values}
        # since we are looking at deck_values: 10, knight, queen, king
        # are valued equally. Update the last element such that we have 4 times
        # as many cards
        self.cards_in_deck[10] = decks*4

    def subtract_card_from_deck(self, card):
        if self.cards_in_deck[card] > 1:
        # if there is more than one card left, subtract it!
            self.cards_in_deck[card] -= 1
        else:
            # if there is exactly one card left, than after it is used we
            # remove the key, thus we cannot draw the card again
            self.cards_in_deck.pop(card)

    def draw_card(self, np_random):
        # we can only draw cards which are in the keys of cards_in_deck.
        card = int(np_random.choice(list(self.cards_in_deck.keys())))
        # subtract the card from the deck
        self.subtract_card_from_deck(card)
        return card

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

    def reset(self, decks):
        self.construct_deck(decks)
        self.dealer = [self.draw_dealer_hand(self.np_random)]
        self.player = self.draw_player_hand(self.np_random)
        return self._get_obs()
