from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *

# load the cloob config
config = CARPConfig.load_yaml("./configs/carp_cloob.yml")
# initialize the cloob model
cloob_model = CARPCloob(config.model)
# load the pretrained model
cloob_model.load("CLOOB CARP Declutr B/")
# cast to GPU
cloob_model = cloob_model.cuda()

# tokenizethe reviews and passages. For more complicated scenarios one could create an online
# data pipeline 
passages = cloob_model.passage_encoder.call_tokenizer("The goose was happy.").to("cuda")
reviews = cloob_model.review_encoder.call_tokenizer(
    ["[quote] the goose is too happy!",
    "[quote] the goose needs to be happier",
    "This story is awful!"]).to("cuda")

# take the encoded passages and reviews and put them into a batch structure.
# in more advanced situations this would let us automate microbatching for inference
# eventually in the future will automatically support multi-GPU inference
passage_batch = BatchElement(passages['input_ids'], passages['attention_mask'])
review_batch = BatchElement(reviews['input_ids'], reviews['attention_mask'])

# Need to use brackets since cosine and calc embeddings are expecting microbatches
with torch.no_grad():
    pass_encs, rev_encs =\
        cloob_model.calculate_embeddings([passage_batch], [review_batch])
    confusion_matrix = cloob_model.cosine_sim(pass_encs[0], rev_encs[0])

# output the confusion matrix of the computation we just did
print(confusion_matrix)