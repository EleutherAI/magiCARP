import torch

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
    p = [passages[ind] for ind in inds]
    r = [reviews[ind] for ind in inds]

    return p, r

def tokenizer_batch(p,  r):
    p_tok = cloob_model.passage_encoder.call_tokenizer(p)
    r_tok = cloob_model.passage_encoder.call_tokenizer(p)
    return p_tok, r_tok

def encode_rand_batch(size):
    p_batch, r_batch = get_random_batch(size)
    p_tok, r_tok = tokenizer_batch(p_batch, r_batch)
    p_batch = BatchElement(p_tok['input_ids'], p_tok['attention_mask'])
    r_batch = BatchElement(r_tok['input_ids'], r_tok['attention_mask'])

    with torch.no_grad():
        pass_encs, rev_encs = \
                cloob_model.calculate_embeddings([p_batch], [r_batch])



