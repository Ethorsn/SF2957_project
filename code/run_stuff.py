import blackjack as bje

if __name__=="__main__":
    env = bje.BlackjackEnvExtend()

    # test class
    class test(object):
        def __init__(self):
            self.ind = 1
        def choice(self, inp):
            return inp[self.ind]
        def getInd(self):
            return self.ind
    # init test class
    t = test()
    print("-- Testing draw_card")
    assert bje.deck[t.getInd()]== bje.draw_card(t)

    print("-- Testing draw_dealer_hand")
    assert 2 == bje.draw_player_hand(t)[t.getInd()]

    print("-- Testing sum_player_hand")
    hand = bje.draw_player_hand(t)
    assert 2*2 == bje.sum_player_hand(hand)

    print("-- Testing player_score not-bust scenario")
    assert 2*2 == bje.player_score(hand)

    print("-- Testing player_score bust scenario")
    hand_mod = hand.copy()
    hand_mod[10] = 2
    assert 0 == bje.player_score(hand_mod)

    print(env.step(1))
