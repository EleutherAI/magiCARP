import torch
import numpy as np
import matplotlib.pyplot as plt

from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.data import *

config = CARPConfig.load_yaml("./vis/carp_cloob.yml")
cloob_model = CARPCloob(config.model)
cloob_model.load("./vis/CLOOB_CP/")

cloob_model = cloob_model.cuda()
pipeline = BaseDataPipeline(path="./carp/dataset")
passages = pipeline.passages
reviews = pipeline.reviews
N = len(passages)
print(N)

def get_random_batch(size):
    inds = torch.randint(0, N, (size,))
    p = [passages[ind][:512] for ind in inds]
    r = [reviews[ind][:512] for ind in inds]

    return p, r

def tokenizer_batch(p,  r):
    p_tok = cloob_model.passage_encoder.call_tokenizer(p)
    r_tok = cloob_model.passage_encoder.call_tokenizer(p)
    return p_tok, r_tok

def encode_rand_batch(size):
    p_batch_s, r_batch_s = get_random_batch(size)
    p_tok, r_tok = tokenizer_batch(p_batch_s, r_batch_s)
    p_batch = BatchElement(p_tok['input_ids'], p_tok['attention_mask'])
    r_batch = BatchElement(r_tok['input_ids'], r_tok['attention_mask'])

    with torch.no_grad():
        pass_encs, rev_encs = \
                cloob_model.calculate_embeddings([p_batch], [r_batch])

    return (pass_encs[0], p_batch_s), (rev_encs[0], r_batch_s)

def encode_and_draw(size):
    _, (pass_encs, pass_batch) = encode_rand_batch(size)
    pass_encs = pass_encs.float()
    U,_,_ = torch.pca_lowrank(pass_encs, q = 2)
    U = U.detach().cpu().numpy()

    # Sort by x value for easier visualization
    U = [[u, pass_] for (u, pass_) in zip(U, pass_batch)]
    U = sorted(U, key = lambda x: x[0][0])
    pass_batch = [x[1] for x in U]
    U = [x[0] for x in U]
    U_x = np.array([u[0] for u in U])
    U_y = np.array([u[1] for u in U])
    for i, pass_ in enumerate(pass_batch):
        print(pass_ + " : " + str(U[i]))

    c = np.random.randint(1,5,size=size)
    names = np.array(pass_batch)
    norm = plt.Normalize(1,4)
    cmap = plt.cm.RdYlGn
    fig, ax = plt.subplots()
    sc = plt.scatter(U_x, U_y, cmap = cmap, norm=norm)
    
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
