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

# Undo spherical coordinate transformation (return to normal domain)
# [n, d - 1] -> [n, d]
def undo_spherical(phi):
    n, d = phi.shape
    d += 1 # d angles makes d + 1 dim vector
    x = torch.ones(n, d, device = phi.device)

    # below forumlas for single [d] x, [d - 1] phi
    # x[0] = cos(phi[0])
    # x[i] = prod(sin(phi[0]), ..., sin(phi[i-1]) * cos(phi[0])
    # x[-1] = prod(sin(phi[0]), ..., sin(phi[-2]) * sin(phi[-1])

    x[:,:d-1] *= torch.cos(phi)

    # make each phi[i] become product of sins of preceeding  elements
    for i in range(d-1):
        crnt = torch.sin(phi[:,i])
        if i == 0:
            phi[:,i] = crnt
        else:
            phi[:,i] = crnt * phi[:,i-1]

    x[:,1:] *= phi

    return x

# assuming [n, d] points (x) are on nsphere
# returns a centroid that is also on nsphere
def nsphere_centroid(x):
    phi = spherical_coord(x) # [n, d - 1]
    mu_phi = phi.mean(0) # [n, d - 1]
    mu = undo_spherical(mu_phi) # [n, d]

    return  mu

# If captions for clusters exist, we can incorporate them
# labels with caption "?" become -1
# multiple labels with same caption get collapsed to first label
def use_captions(labels):
    with open("metalabel_data/user_captions.txt") as file:
        lines = file.readlines()
    lines = [line[:-1] for line in lines] # remove the \n at the end
    kvps = [[i, lines[i]] for i in range(len(lines))]

    # first get rid of the labels that dont have captions (i.e. ?)
    inds = []
    for i in range(len(kvps)):
        if kvps[i][1] == "?":
            inds.append(i)

    kvps = [[-1 if i in inds else i, kvps[i][1]] for i in range(len(kvps))]

    # now gather labels that have multiple captions
    ind_groups = []
    processed = []
    for i in range(len(kvps)):
        if kvps[0] == "-1": continue
        if i in processed: continue

        group = [i]

        # check all preceeding kvps for same caption
        for j in range(i + 1, len(kvps)):
            if kvps[j][1] == kvps[i][1]:
                group.append(j)
                processed.append(j)
        ind_groups.append(group)

    # ind groups is now list of groups of indices
    # a single group represents multiple labels with same caption
    
    # now for each group, make labels the same as the first in the group
    for group in ind_groups:
        to_set = kvps[group[0]][0] # the label we want to set everything else to
        for ind in group[1:]:
            labels[labels == ind] = to_set
            kvps[ind][0] = to_set

    # now we have kvps as a map for the labels
    # key is what we want to map to, position is the relevant label
    map_ = { i : kvps[i][0] for i in range(len(kvps)) }
    map_[-1] = -1 # -1 should still map to itself

    labels = np.vectorize(map_.get)(labels)

    kvps = list(set(map(tuple, kvps))) # safe to remove duplicates now
    kvps.sort(key = lambda x: x[0]) # sort by label

    # now lets rescale labels to be [0, n_labels - 1]
    unique_vals = np.unique(labels)

    # map to [-1, 0, ... ,n]
    map_ = {unique_vals[i] : i - 1 for i in range(len(unique_vals))}


    labels = np.vectorize(map_.get)(labels) # convert labels to new range

    captions = [kvp[1] for kvp in kvps] # recover captions for new label set
    captions = captions[1:] # exclude error label

    return labels, captions
