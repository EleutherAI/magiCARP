import torch
import numpy as np
from transformers import pipeline

# Get sentiment scores for all strings in names_sub (a list of strings)
def extract_sentiment(names_sub):
    pipe = pipeline("sentiment-analysis", device = 0)
    # batch the input
    inds = torch.arange(len(names_sub))
    batch_size = 32
    inds_list = inds.chunk(len(inds) // batch_size)

    sents = []
    for inds in inds_list:
        name_batch = [names_sub[ind] for ind in inds]
        pipe_in = pipe.preprocess(name_batch, truncation = True, padding = True)
        sentiments = pipe.forward(pipe_in) # list of dicts, each has keys 'label' (v: STR), 'score' (v: float)
        sentiments = sentiments[0].cpu().numpy() # -> [n, 2] logits 
        sents.append(sentiments)
    sents = np.concatenate(sents)
    sentiments = sents
    # sanity
    assert len(sents) == len(names_sub)

    # sentiments are logits, but
    # we want: 0 - strong negative, 1 - strong positive and a gradient between the two
    def get_sent_scores(sentiments):
        neg = sentiments[:,0]
        pos = sentiments[:,1]
        scores = pos - neg
        scores -= scores.min()
        scores /= scores.max()

        return scores

    return get_sent_scores(sents)

# given names (list of strings) returns indices of all instances
# that pass predicate on sentiment
def filter_on_sentiment(names, pred = lambda x : x < 0.8):
    scores = extract_sentiment(names)
    return np.where(pred(scores))[0]