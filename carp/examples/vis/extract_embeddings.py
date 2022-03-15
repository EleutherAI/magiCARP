import torch
import numpy as np
from torch.nn.functional import normalize
import math

from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.data import *

CHUNK_SIZE = 512
MAX_CHUNKS = -1 # Only embed this many chunks (set to -1 for all)
N_CTX = 512
OFFSET = 0 # skip to this sample

# Lots of stuff from before now encapsulated in this loader class
# Load model
print("Load Model")
config = CARPConfig.load_yaml("configs/carp_cloob.yml")
cloob_model = CARPCloob(config.model)
cloob_model.load("checkpoints/CLOOB_CP/")
cloob_model = cloob_model.cuda()

# Loader to forward pass and embed embeddings
LOADER_IND = 0
class Loader:
    def __init__(self):
        pipeline = BaseDataPipeline(path="carp/dataset")
        self.passages = pipeline.passages
        self.reviews = pipeline.reviews
        
        self.N = len(self.passages) # assume this is equal to len(reviews)
        self.ind = LOADER_IND
        
        self.preproc()
        #self.shuffle()
    
    def preproc(self):
        self.passages = [passage[-N_CTX:] for passage in self.passages]
        self.reviews = [review[-N_CTX:] for review in self.reviews]

    def postproc(self, x):
        x = normalize(x[0].float())
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

# Allocate tensors to store all passage and rev encodings
LATENT_DIM = 2048
P = torch.zeros(loader.N, LATENT_DIM, dtype = torch.half)
R = torch.zeros(loader.N, LATENT_DIM, dtype = torch.half)

# To be safe I add a checkpoint feature

checkpoint_interval = 100

def save():
    torch.save(P, "passage_encs.pt")
    torch.save(R, "review_encs.pt")

def load():
    try:
        P = torch.load("passage_encs.pt")
        R = torch.load("review_encs.pt")

        # find first non zero row to set offset
        for i, x in enumerate(P):
            if x.pow(2).sum() < 0.5:
                OFFSET = i
                break
        
        print("Found {} encodings. Continuing on this.".format(OFFSET))
    except:
        print("Couldn't Find Existing Encodings, Starting From Scratch")

load()
# set offset to loader
i = OFFSET
LOADER_IND = (OFFSET // CHUNK_SIZE)

print("Load Dataset")
loader = Loader()

from tqdm import tqdm

print(P[0])
# on 3090 ~45 minutes to do all
print("Embedding Chunks")
for (pass_encs, rev_encs) in tqdm(loader):
    print(pass_encs[0])
    exit()
    P[i:i + CHUNK_SIZE] = pass_encs
    R[i:i + CHUNK_SIZE] = rev_encs

    i+=CHUNK_SIZE
    #if (i//CHUNK_SIZE) == checkpoint_interval: save()
    if (i//CHUNK_SIZE) >= MAX_CHUNKS and MAX_CHUNKS != -1: break 