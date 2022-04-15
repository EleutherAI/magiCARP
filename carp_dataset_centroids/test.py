import torch
import pandas as pd

pos = torch.load('dataset_centroid_dists_pos.pt')
print(pos.shape)
pos = pos.type(torch.float32)
print(pos[0])

softmaxed_positives = torch.softmax(pos, dim=1)
print(softmaxed_positives.shape)
print(torch.sum(softmaxed_positives[0]))
print(softmaxed_positives[0])

argmaxes = torch.max(softmaxed_positives, dim=-1).values
res = argmaxes >= .01
filtered_softmax = softmaxed_positives[res]
print(filtered_softmax.shape)

passages = pd.read_csv('dataset.csv')['passages']


metalabels_pd = pd.read_csv('passage_metalabel_dataset.csv')
passages = list(metalabels_pd["passages"])
# get the target distributions
reviews = torch.tensor(metalabels_pd[[str(i) for i in range(92)]].values.tolist())
print(reviews.shape)
print(reviews[0])
filtered_reviews = reviews[torch.max(reviews, dim=-1).values > .5]
print(filtered_reviews.shape)
