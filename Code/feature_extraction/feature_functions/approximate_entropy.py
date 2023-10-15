import EntropyHub as EH

"""Approximate Entropy"""
def approximate_entropy(sig, step_number):
    wind = int(step_number/5)
    m = int(len(sig)/wind)
    r = 0.2 * sig.std()  #perché???
    apen, log = EH.ApEn(list(sig), m=m, r=r)
    return apen.mean()