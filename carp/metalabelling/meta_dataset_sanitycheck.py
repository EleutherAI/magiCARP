import torch
import numpy as np
from torch.nn.functional import normalize
import math

from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *
from carp.pytorch.data import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# check if centroids have been created
try:
    centroids = torch.load("metalabel_data/centroids.pt").half() # [2, k, d], pos and neg centroids
except:
    print("One or more pre-req files were not found")
    print("Ensure you run carp.metalabelling.get_centroids")

# get classifier distribution from centroids
# gets cosine similarity for each sample with each centroid
def get_cos_sims(review_encs, centroids):
# review_encs: [n, d]
# centroids: [k, d]
    review_encs = review_encs.float().to(device) # cuda speeds up the matmult
    centroids = centroids.float().to(device)

    centroid_norms = torch.norm(centroids, dim = 1) # [k]
    review_norms = torch.norm(review_encs, dim = 1) # [n]

    norm_scale = centroid_norms[None,:] * review_norms[:,None] # [n, k]

    cos_sims = review_encs @ centroids.t() / norm_scale # [n, k]

    return cos_sims.cpu().half()

# get indices of top n values in tensor and the values themselves
def get_top_inds(tensor, n):
    top_vals, top_inds = torch.topk(tensor.float(), n)
    # convert both to lists
    top_vals = top_vals.cpu().numpy().tolist()
    top_inds = top_inds.cpu().numpy().tolist()
    return list(zip(top_vals, top_inds))

if __name__ == "__main__":
    print("Load Model")
    config = CARPConfig.load_yaml("configs/carp_cloob.yml")
    cloob_model = CARPCloob(config.model)
    cloob_model.load("checkpoints/CLOOB_CP/")
    cloob_model = cloob_model.to(device)

    N_CTX = 512

    tokenize = cloob_model.passage_encoder.call_tokenizer

    def encode(txt_batch, mode = "R"):
        tok_out = tokenize(txt_batch)
        x = tok_out["input_ids"]
        mask = tok_out["attention_mask"]
        enc_input = BatchElement(x, mask)

        with torch.no_grad():
            if mode == "R":
                encs = cloob_model.encode_reviews(enc_input).hidden
            if mode == "P":
                encs = cloob_model.encode_passages(enc_input).hidden

        encs = encs.cpu().half()
        return encs

    print("Usage: Type MODE (R/P) followed by a space and the text to encode")
    while True:
        txt = input()
        if txt == "exit":
            break
        mode = txt[0]
        txt = txt[2:]

        if mode == "r" or mode == "R":
            encs = encode([txt], mode = "R")
        elif mode == "p" or mode == "P":
            encs = encode([txt], mode = "P")
        print(encs.shape) # (1, d)

        # get cosine similarity for positive centroids
        cos_sims = get_cos_sims(encs, centroids[1]).squeeze()
        print("Positive Classification: {}".format(cos_sims.argmax().item()))
        print(get_top_inds(cos_sims, 5))

        # get cosine similarity for negative centroids
        cos_sims = get_cos_sims(encs, centroids[2]).squeeze()
        print("Negative Classification: {}".format(cos_sims.argmax().item()))
        print(get_top_inds(cos_sims, 5))

