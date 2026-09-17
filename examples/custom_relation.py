"""Edit f to test your own deterministic relation. This is evaluation only."""


def f(a, b):
    return a // b


def valid(a, b):
    # Optional filter for undefined inputs; remove for functions valid everywhere.
    return b != 0
