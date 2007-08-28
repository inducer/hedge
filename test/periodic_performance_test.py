"""This benchmark informs the decision whether to do matching of periodic
face pairs by lookup of tuples of vertices or each vertex separately.

At the time of this writing, and under Python 2.5, tuplewise lookup has
about a 1-to-4 advantage.
"""

def main():
    from random import choice
    from time import time

    N = 50
    base_set = range(N)
    
    # generate correspondence
    corr = dict((b, choice(base_set)) for b in base_set)

    # generate tuples
    TS = 3
    N_TUP = 500000
    tups = [tuple(choice(base_set) for i in range(TS))
            for j in range(N_TUP)]

    tup_corr = {}
    for tup in tups:
        mapped = tuple(corr[t] for t in tup)
        tup_corr[tup] = mapped

    ITER = 500000
    start = time()
    for i in xrange(ITER):
        tup = choice(tups)
        mapped = tuple(corr[t] for t in tup)
    print "elwise", time()-start

    start = time()
    for i in xrange(ITER):
        tup = choice(tups)
        mapped = tup_corr[tup]
    print "tuplewise", time()-start

if __name__ == "__main__":
    main()
