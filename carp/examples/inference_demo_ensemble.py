from functools import reduce

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from carp.configs import CARPConfig
from carp.pytorch.model.architectures import *

# load the cloob config
config = CARPConfig.load_yaml("./configs/carp_l.yml")
# initialize the cloob model
carp_model = CARP(config.model)
# load the pretrained model
carp_model.load("CARP_L_new/")
# cast to GPU
carp_model = carp_model.cuda()

# tokenizethe reviews and passages. For more complicated scenarios one could create an online
# data pipeline
passages = carp_model.passage_encoder.call_tokenizer(
    [
        "The goose is very happy. He finally could eat his favourite cookies.",
        "The goose is very sad. He finally could eat his favourite cookies.",
    ]
).to("cuda")
reviews = [
    "The goose is too happy, he should be sadder!",
    "The goose is too sad, he should be happier!",
    "this story does not make sense!",
    "this story is awful!",
]

review_count = len(reviews)

# soften reviews using paraphrasing (allows for CARP based boosting)
tokenizer_pegasus = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
model_pegasus = (
    PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
    .half()
    .to("cuda")
)

# We'll end up with an ensemble of size 6 per review
num_beams = 10


# Paraphrases using peagasus.
def get_review_ensemble(input_text):
    batch = tokenizer_pegasus(
        input_text,
        truncation=True,
        padding="longest",
        max_length=60,
        return_tensors="pt",
    ).to("cuda")
    translated = model_pegasus.generate(
        **batch,
        max_length=60,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        temperature=1.5
    )
    return tokenizer_pegasus.batch_decode(translated, skip_special_tokens=True)


# append quote tokens, doubling the number of reviews. tokenize
reviews = carp_model.review_encoder.call_tokenizer(
    reduce(
        lambda a, b: a + [b],
        list(map(lambda x: "[quote] " + x, get_review_ensemble(reviews))),
        [],
    )
).to("cuda")

# take the encoded passages and reviews and put them into a batch structure.
# in more advanced situations this would let us automate microbatching for inference
# eventually in the future will automatically support multi-GPU inference
passage_batch = BatchElement(passages["input_ids"], passages["attention_mask"])
review_batch = BatchElement(reviews["input_ids"], reviews["attention_mask"])

# Need to use brackets since cosine and calc embeddings are expecting microbatches
with torch.no_grad():
    pass_encs, rev_encs = carp_model.calculate_embeddings(
        [passage_batch], [review_batch]
    )
    # compute cosine similarity and average over the last dimension
    confusion_matrix = torch.mean(
        carp_model.cosine_sim(pass_encs[0], rev_encs[0]).reshape(
            passages.input_ids.shape[0], review_count, num_beams
        ),
        dim=-1,
    )

# output the confusion matrix of the computation we just did
print(confusion_matrix)
