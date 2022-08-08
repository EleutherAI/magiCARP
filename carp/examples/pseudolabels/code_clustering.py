import torch
import numpy as np
import random

from hdbscan import HDBSCAN
from umap import UMAP

import joblib

from carp.pytorch.data import CodeReviewPipeline
from carp.pytorch.model.architectures.carp import CARP
from carp.configs import CARPConfig

from carp.examples.encodings.encode_reviews import enc_reviews
from carp.examples.encodings.encode_passages import enc_passages
from carp.examples.visualization.plot_util import scatter_with_names

base_tform = lambda x: " ".join(x)

if __name__ == "__main__":
    do_reviews = True
    print("Load Dataset...")
    pipeline = CodeReviewPipeline()

    # Try loading all components in case they were already generated
    try:
        encs = torch.load("carp/examples/pseudolabels/encs.pt")
        inds = torch.load("carp/examples/pseudolabels/embedding_inds.pt")
        umap_tform = joblib.load("carp/examples/pseudolabels/umap_tform.joblib")
        hdbscan_clusterer = joblib.load("carp/examples/pseudolabels/hdbscan_clusterer.joblib")
        labels = np.load("carp/examples/pseudolabels/cluster_labels.npy")

        encs = umap_tform.transform(encs)
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Load Model...")
        config = CARPConfig.load_yaml("configs/codecarp.yml")
        model = CARP(config.model)
        model.load("checkpoints/CodeCARPv2/1000/")
        model = model.to(device)

        enc_fn = enc_reviews if do_reviews else enc_passages

        print("Generate Encodings...")
        enc_fn(
            N_SAMPLES = 10000,
            force_fresh = True,
            CHUNK_SIZE = 128,
            SAVE_EVERY = 5,
            model = model,
            txt_data = pipeline.reviews if do_reviews else pipeline.passages,
            ind_path = "carp/examples/pseudolabels/embedding_inds.pt",
            enc_path = "carp/examples/pseudolabels/encs.pt",
            random_state = 69,
            tform = base_tform
        )

        encs = torch.load("carp/examples/pseudolabels/encs.pt").float()
        inds = torch.load("carp/examples/pseudolabels/embedding_inds.pt")

        print("Performing UMAP...")
        umap_tform = UMAP(
            n_neighbors = 30,
            min_dist = 0.0,
            n_components = 2,
            metric = 'cosine',
            random_state = 42,
            low_memory = False
        ).fit(encs)

        joblib.dump(umap_tform, "carp/examples/pseudolabels/umap_tform.joblib")
        encs = umap_tform.transform(encs)

        print("Performing HDBSCAN Clustering...")
        hdbscan_clusterer = HDBSCAN(
            min_samples = 5,
            min_cluster_size = 30,
            prediction_data = True
        ).fit(encs)

        joblib.dump(hdbscan_clusterer, "carp/examples/pseudolabels/hdbscan_clusterer.joblib")
        labels = hdbscan_clusterer.labels_
        np.save("carp/examples/pseudolabels/cluster_labels.npy", labels)
    
    # Get corresponding reviews from dataset and append label to front of each
    _txt = [(pipeline.reviews if do_reviews else pipeline.passages)[ind] for ind in inds]
    _txt = [str(label) + " " + base_tform(rev) for (label, rev) in zip(labels, _txt)]

    # Create the plot
    x = encs[:,0]
    y = encs[:,1]
    c = np.where(labels != -1, labels / labels.max(), labels) # normalize non error labels to [0, 1] for color
    scatter_with_names(x, y, _txt, c = c)








            
