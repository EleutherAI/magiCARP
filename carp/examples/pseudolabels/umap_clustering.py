import torch
import numpy as np
import random

from hdbscan import HDBSCAN
from umap import UMAP

import joblib

from carp.pytorch.data import *
from carp.pytorch.model.architectures.carp import CARP
from carp.configs import CARPConfig

from carp.examples.encodings.encode_reviews import enc_reviews
from carp.examples.visualization.plot_util import scatter_with_names

if __name__ == "__main__":
    print("Load Dataset...")
    pipeline = BaseDataPipeline(path="carp/dataset")

    # Try loading all components in case they were already generated
    try:
        review_encs = torch.load("carp/examples/pseudolabels/review_encs.pt")
        inds = torch.load("carp/examples/pseudolabels/embedding_inds.pt")
        umap_tform = joblib.load("carp/examples/pseudolabels/umap_tform.joblib")
        hdbscan_clusterer = joblib.load("carp/examples/pseudolabels/hdbscan_clusterer.joblib")
        labels = np.load("carp/examples/pseudolabels/cluster_labels.npy")

        review_encs = umap_tform.transform(review_encs)
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use CARP Large by default
        print("Load Model...")
        config = CARPConfig.load_yaml("configs/carp_l.yml")
        model = CARP(config.model)
        model.load("checkpoints/CARP_L")
        model = model.to(device)

        print("Generate Encodings...")
        enc_reviews(
            N_SAMPLES = 10000,
            force_fresh = True,
            CHUNK_SIZE = 128,
            SAVE_EVERY = 5,
            model = model,
            txt_data = pipeline.reviews,
            ind_path = "carp/examples/pseudolabels/embedding_inds.pt",
            enc_path = "carp/examples/pseudolabels/review_encs.pt",
            random_state = 69,
        )

        review_encs = torch.load("carp/examples/pseudolabels/review_encs.pt").float()
        inds = torch.load("carp/examples/pseudolabels/embedding_inds.pt")

        print("Performing UMAP...")
        umap_tform = UMAP(
            n_neighbors = 30,
            min_dist = 0.0,
            n_components = 2,
            metric = 'cosine',
            random_state = 42,
            low_memory = False
        ).fit(review_encs)

        joblib.dump(umap_tform, "carp/examples/pseudolabels/umap_tform.joblib")
        review_encs = umap_tform.transform(review_encs)

        print("Performing HDBSCAN Clustering...")
        hdbscan_clusterer = HDBSCAN(
            min_samples = 5,
            min_cluster_size = 30,
            prediction_data = True
        ).fit(review_encs)

        joblib.dump(hdbscan_clusterer, "carp/examples/pseudolabels/hdbscan_clusterer.joblib")
        labels = hdbscan_clusterer.labels_
        np.save("carp/examples/pseudolabels/cluster_labels.npy", labels)
    
    # Get corresponding reviews from dataset and append label to front of each
    reviews_txt = [pipeline.reviews[ind] for ind in inds]
    reviews_txt = [str(label) + " " + rev for (label, rev) in zip(labels, reviews_txt)]

    # Create the plot
    x = review_encs[:,0]
    y = review_encs[:,1]
    c = np.where(labels != -1, labels / labels.max(), labels) # normalize non error labels to [0, 1] for color
    scatter_with_names(x, y, reviews_txt, c = c)








            
