import EntropyHub as EH

"""Approximate Entropy"""
def approximate_entropy(sig, step_length):
    m = int(len(sig) / step_length)
    r = 0.2 * sig.std()  #perché???
    apen, log = EH.ApEn(list(sig), m=m, r=r)
    return apen.mean()