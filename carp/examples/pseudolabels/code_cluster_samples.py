import torch
import numpy as np
import random

import joblib
from typing import Iterable, Tuple

from carp.examples.pseudolabels.clustering_util import cull
from carp.pytorch.data import *

base_tform = lambda x: " ".join(x)

if __name__ == "__main__":
    print("Load Dataset...")
    pipeline = CodeReviewPipeline()

    # load labels and indices of reviews they correspond to
    try:
        inds = torch.load("carp/examples/pseudolabels/embedding_inds.pt")
        labels = np.load("carp/examples/pseudolabels/cluster_labels.npy")
    except:
        print("Could not find nessecary objects. Please ensure you have run code_clustering.py")
        exit()

    review_txt = [pipeline.reviews[ind] for ind in inds]
    
    # remove error labels
    cull_inds = cull(labels, -1)
    labels = labels[cull_inds]
    review_txt = [base_tform(review_txt[i]) for i in cull_inds]

    # given a list and indices for that list, returns:
    # - list with those indices removed
    # - list with values from those indices in order
    def split_list(L : Iterable[any], inds : Iterable[int]) -> Tuple[Iterable[any], Iterable[any]]:
        L_not = []
        # popping in reverse keeps pops from messing up indexing of earlier elements
        for i in reversed(inds):
            L_not.append(L.pop(i))
        L_not.reverse()

        return L, L_not

    # assumes labels are 0 thru labels.max()
    for i in range(labels.max() + 1):
        # indices of everything to keep
        cull_inds = cull(labels, i) 
        labels = labels[cull_inds]

        # [reviews kept after cull, reviews removed by cull]
        reviews_i, review_txt = split_list(review_txt, cull_inds)
        
        # shuffle then print first 10 samples
        random.shuffle(reviews_i)
        print("================")
        print("Reviews With Label {}:".format(i))

        for j in range(10):
            print(reviews_i[j])