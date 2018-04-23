import gym.envs.toy_text.blackjack as bj
import gym.spaces as spaces

class bj2(bj.BlackjackEnv):
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
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}


def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

def reset(self):
    self.dealer = draw_hand(self.np_random)
    self.player = draw_hand(self.np_random)
    return self._get_obs()
