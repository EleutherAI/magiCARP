import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import normalize

from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.data import *


# Load model
config = CARPConfig.load_yaml("./vis/carp_cloob.yml")
cloob_model = CARPCloob(config.model)
cloob_model.load("./vis/CLOOB_CP/")
cloob_model = cloob_model.cuda()

# Extract raw reviews and passages from pipeline
pipeline = BaseDataPipeline(path="./carp/dataset")
passages = pipeline.passages
reviews = pipeline.reviews
N = len(passages)
print(N)

# Sample random batch from dataset
def get_random_batch(size):
    inds = torch.randint(0, N, (size,))
    p = [passages[ind][:512] for ind in inds]
    r = [reviews[ind][:512] for ind in inds]

    return p, r

# Use tokenizer on a batch of passages and reviews
def tokenizer_batch(p,  r):
    p_tok = cloob_model.passage_encoder.call_tokenizer(p)
    r_tok = cloob_model.passage_encoder.call_tokenizer(p)
    return p_tok, r_tok
    
# construct batch element from a tokenizer output
def make_batch_elem(tok_out):
    return BatchElement(tok_out['input_ids'],
                       tok_out['attention_mask'])

# Sample random batch from dataset and encode it
def encode_rand_batch(size):
    p_batch_s, r_batch_s = get_random_batch(size)
    p_tok, r_tok = tokenizer_batch(p_batch_s, r_batch_s)
    p_batch = make_batch_elem(p_tok)
    r_batch = make_batch_elem(r_tok)

    with torch.no_grad():
        pass_encs, rev_encs = \
                cloob_model.calculate_embeddings([p_batch], [r_batch])
    # -> [[pass_encs], [rev_encs]]
    pass_encs = pass_encs[0].float()
    rev_encs = rev_encs[0].float()
    
    pass_encs = normalize(pass_encs)
    rev_encs = normalize(rev_encs)

    return (pass_encs, p_batch_s), \
        (rev_encs, r_batch_s)

# Given encodings and batch (strings), return PCA result
# and batch sorted by first value in PCA vectors
def pca_sort(encs, batch):
    U,_,_ = torch.pca_lowrank(encs, q = 2)
    U = U.detach().cpu().numpy()

    # Sort by x value for easier visualization
    U = [[u, pass_] for (u, pass_) in zip(U, batch)]
    U = sorted(U, key = lambda x: x[0][0])
    batch = [x[1] for x in U]

    # Get 2d vectors from PCA
    U = [x[0] for x in U]

    return U, batch

# scatter plot with labels/names
# x = array like
# y = array like same length as x
# names = list of strings
# source: https://stackoverflow.com/q/7908636
def scatter_with_names(x, y, names):
    c = np.random.randint(1,5,size=len(x))
    names = np.array(names)
    norm = plt.Normalize(1,4)
    cmap = plt.cm.RdYlGn
    fig, ax = plt.subplots()
    sc = plt.scatter(x, y, cmap = cmap, norm=norm)
    
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                               " ".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()   
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()

# Mode = "R" or "P" review or passage
def encode_and_draw(size, mode = "R"):
    if mode == "R":
        _, (encs, batch) = encode_rand_batch(size)
    else:
        (encs, batch), _ = encode_rand_batch(size)
    
    U, batch = pca_sort(encs, batch)
    U_x = np.array([u[0] for u in U])
    U_y = np.array([u[1] for u in U])
    
    scatter_with_names(U_x, U_y, batch)

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

# Mode = "R" or "P" review or passage
def spherecoord_ead(size, mode = "R"):
    if mode == "R":
        _, (encs, batch) = encode_rand_batch(size)
    else:
        (encs, batch), _ = encode_rand_batch(size)
    encs = spherical_coord(encs)
    
    U, batch = pca_sort(encs, batch)
    U_x = np.array([u[0] for u in U])
    U_y = np.array([u[1] for u in U])
    
    scatter_with_names(U_x, U_y, batch)

# Compare sphere to normal
def spherebasecompare(size, mode = "R"):
    if mode == "R":
        _, (encs, batch) = encode_rand_batch(size)
    else:
        (encs, batch), _ = encode_rand_batch(size)
    
    sphere_encs = spherical_coord(encs)

    U_sphere, batch_sphere = pca_sort(sphere_encs, batch)
    U, batch = pca_sort(encs, batch)

    U_x = np.array([u[0] for u in U])
    U_y = np.array([u[1] for u in U])
    
    scatter_with_names(U_x, U_y, batch)

    U_x = np.array([u[0] for u in U_sphere])
    U_y = np.array([u[1] for u in U_sphere])

    scatter_with_names(U_x, U_y, batch_sphere)

# Main functions of interest here are:
# encode_and_draw(size, mode) which makes scatter plot of PCA embeddings
# spherecoord_ead(size, mode) which does the same but with spherical coordinates of embeddings
# spherebasecompare(size, mode) does both one after the other for direct comparison
if __name__ == "__main__":
    spherebasecompare(1024, mode = "R")
    