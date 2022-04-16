import torch
from torch import nn
import numpy as np

# chunk list into chunks of size n
def chunk(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

# cosine similarity matrix (outer product)
def cos_sims(a, b, device):
    # a: [n, d], b: [k, d]
    a = a.float().to(device)
    b = b.float().to(device)

    a_norms = torch.norm(a, dim = 1) # [n]
    b_norms = torch.norm(b, dim = 1) # [k]
    scale = a_norms[:,None] * b_norms[None,:] # [n, k]

    b = b.t() # [d, k]

    res = a @ b / scale
    return res

# Return list of indices of all elements that are not a given label
def cull(labels, to_cull):
    return np.where(labels != to_cull)[0].tolist()

# Parameterize x as angles of a unit hypersphere
# Further explanation in notebook
# assume x is normalized (l2 norm x_i = 1)
# assumes d >= 3
def spherical_coord(x):
    n, d = x.shape
    # -> unit n-sphere points

    # phi[:,0:d-2] in [0, pi]
    # phi[:,d-2] in [-pi, pi]
    phi = torch.zeros_like(x)[:,:-1] # -> (n, d-1)

    # value being curried to compute next angle
    curr = x[:,d-1].pow(2) + x[:,d-2].pow(2)
    
    # compute last angle first (it has weird formula conditioned on x_d)
    phi[:,d-2] = torch.acos(x[:,d-2] / curr.sqrt())
    phi[:,d-2] = torch.where(x[:,d-1] >= 0, phi[:,d-2], 2 * np.pi - phi[:,d-2])
    
    # compute the rest
    for i in reversed(range(0, d-2)):
        curr += x[:,i].pow(2)
        phi[:,i] = torch.acos(x[:,i] / curr.sqrt())

    return phi