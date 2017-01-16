import random
import parser

def disturb_output(Xy, noise_rate):
    Xy = Xy.copy()
    y = Xy[:,-1]
    N = len(y)

    labels = set(y)
    idx = random.sample(range(N),int(N*noise_rate))

    for i in idx:
        other_labels = labels.difference(set([y[i]]))
        Xy[i,-1] = random.sample(list(other_labels),1)[0]

    return Xy
