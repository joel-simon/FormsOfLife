import random, string

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def rand_key():
    return ''.join(random.choice(string.ascii_uppercase) for i in range(5))