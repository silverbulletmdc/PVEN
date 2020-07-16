from .tuplet_loss import *

def test_generate_tuplets():
    print(generate_tuplets(4, 8))
    print(generate_tuplets(5, 8))


def test_tuplet_loss():
    K = 4
    P = 8
    s = 32
    C = 16
    tuplet_loss = TupletLoss(K, P, s)
    feats = torch.randn(K * P, C)
    loss = tuplet_loss(feats)
    print(loss)
