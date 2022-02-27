import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import normalize

from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.data import *

from carp.examples.vis.vis_util import spherical_coord, scatter_with_names

CHUNK_SIZE = 512
MAX_CHUNKS = 16 # Only embed this many chunks (set to -1 for all)
N_CTX = 512
SPHERICAL = True

# Lots of stuff from before now encapsulated in this loader class
# Load model
print("Load Model")
config = CARPConfig.load_yaml("configs/carp_cloob.yml")
cloob_model = CARPCloob(config.model)
cloob_model.load("checkpoints/CLOOB_CP/")
cloob_model = cloob_model.cuda()

# Loader to forward pass and embed embeddings
class Loader:
    def __init__(self):
        pipeline = BaseDataPipeline(path="carp/dataset")
        self.passages = pipeline.passages
        self.reviews = pipeline.reviews
        
        self.N = len(self.passages) # assume this is equal to len(reviews)
        self.ind = 0
        
        self.preproc()
        self.shuffle()
    
    def preproc(self):
        self.passages = [passage[-N_CTX:] for passage in self.passages]
        self.reviews = [review[-N_CTX:] for review in self.reviews]

    def postproc(self, x):
        x = normalize(x[0].float())
        if SPHERICAL: x = spherical_coord(x)
        x = x.cpu().detach().half()
        return x
        
    def shuffle(self):
        inds = torch.randperm(self.N)
        self.passages = [self.passages[ind] for ind in inds]
        self.reviews = [self.reviews[ind] for ind in inds]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Get batch
        passages = self.passages[self.ind * CHUNK_SIZE : (self.ind+1) * CHUNK_SIZE]
        reviews = self.reviews[self.ind * CHUNK_SIZE : (self.ind+1) * CHUNK_SIZE]
        
        # Tokenize
        tokenize = cloob_model.passage_encoder.call_tokenizer
        p_tok_out = tokenize(passages)
        r_tok_out = tokenize(reviews)
        p_input_ids = p_tok_out["input_ids"]
        p_masks = p_tok_out["attention_mask"]
        r_input_ids = r_tok_out["input_ids"]
        r_masks = r_tok_out["attention_mask"]
        
        p_input = BatchElement(p_input_ids, p_masks)
        r_input = BatchElement(r_input_ids, r_masks)
        
        with torch.no_grad():
            pass_encs, rev_encs = \
                cloob_model.calculate_embeddings([p_input], [r_input])
        
        pass_encs = self.postproc(pass_encs)
        rev_encs = self.postproc(rev_encs)
        
        self.ind += 1
        
        return pass_encs, rev_encs
    
    next = __next__

print("Load Dataset")
loader = Loader()

# Allocate tensors to store all passage and rev encodings
LATENT_DIM = 2047 if SPHERICAL else 2048
P = torch.zeros(loader.N, LATENT_DIM, dtype = torch.half)
R = torch.zeros(loader.N, LATENT_DIM, dtype = torch.half)

from tqdm import tqdm

# on 3090 ~45 minutes to do all
i = 0
print("Embedding Chunks")
for (pass_encs, rev_encs) in tqdm(loader):
    P[i:i + CHUNK_SIZE] = pass_encs
    R[i:i + CHUNK_SIZE] = rev_encs

    i+=CHUNK_SIZE
    if (i//CHUNK_SIZE) >= MAX_CHUNKS and MAX_CHUNKS != -1: break 

P = P[:i]
R = R[:i]

P_names = loader.passages[:i]
R_names = loader.reviews[:i]

# UMAP stuff
import umap
reducer = umap.UMAP()

print("UMAP Reducing Dimensions of Embeddings")
P = reducer.fit_transform(P)
R = reducer.fit_transform(R)

print("Plotting")
x = np.concatenate((P[:,0], R[:,0]))
y = np.concatenate((P[:,1], R[:,1]))
names = P_names + R_names

scatter_with_names(x, y, names)
