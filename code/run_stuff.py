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

    print("exit 0")
