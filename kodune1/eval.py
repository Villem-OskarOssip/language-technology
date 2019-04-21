"""
Estimates the accuracy of a gap test solution.
Reads in a stream of ranks from the stdin and 
outputs accuracy at levels 1-10 to the stdout.

Usage example:

    $ python example_test.py <gap file> <candidates file> > python eval.py
    $ 0.4,0.6,0.9,1.0,1.0,1.0,1.0,1.0,1.0,1.0

"""
import sys


if __name__ == '__main__':
    ranks = [0] * 10
    for n, ln in enumerate(sys.stdin):
        rank = int(ln.rstrip())
        if rank < 10:
            ranks[rank] += 1
    n = float(n+1)
    for i in range(1, 10):
        ranks[i] += ranks[i-1]
    acc = [r / n for r in ranks]
    print(','.join('%.2f' % a for a in acc))
