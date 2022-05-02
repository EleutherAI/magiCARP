import torch


# load review encodings
def load_encs(path):
    # remove trailing zero vectors
    def remove_zeros(T):
        bound = len(T)
        for i in reversed(range(len(T))):
            if T[i].sum() > 0.25:  # i.e. not 0
                bound = i + 1
                break
        print("{} encodings found.".format(bound))
        return T[:bound]

    print("Loading Encodings...")
    review_encs = torch.load(path)

    return remove_zeros(review_encs)


# Save encodings
def save_encs(encs, path):
    torch.save(encs, path)


# chunk iterable into chunks of size n
def chunk(l, n):
    return [l[i : i + n] for i in range(0, len(l), n)]
