
import torch
import numpy as np
import random

from hdbscan import HDBSCAN
from umap import UMAP

import joblib

from carp.pytorch.data import *
from carp.pytorch.model.architectures.carp_direct import CARPDirect
from carp.configs import CARPConfig
from carp.examples.magiplots.sentiment import extract_sentiment

from carp.examples.encodings.encode_reviews import enc_reviews

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load dataset
    print("Load Dataset...")
    pipeline = BaseDataPipeline(path="carp/dataset")
    reviews = pipeline.reviews[:-1000] # dont take validation set from training

    # ===== UMAP EMBEDDINGS AND CLUSTERS =====

    # Try loading all components in case they were already generated
    try:
        review_encs = torch.load("carp/examples/magiplots/review_encs.pt")
        inds = torch.load("carp/examples/magiplots/embedding_inds.pt")
        umap_tform = joblib.load("carp/examples/magiplots/umap_tform.joblib")
        hdbscan_clusterer = joblib.load("carp/examples/magiplots/hdbscan_clusterer.joblib")
        labels = np.load("carp/examples/magiplots/cluster_labels.npy")

        review_encs = umap_tform.transform(review_encs)

        scores = np.load("carp/examples/magiplots/")
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use CARP Large by default
        print("Load Model...")
        config = CARPConfig.load_yaml("configs/carp_declutr.yml")
        model = CARPDirect(config.model)
        model.load("checkpoints/CARP_DECLUTR/")
        model = model.to(device)

        print("Generate Encodings...")
        enc_reviews(
            N_SAMPLES = 10000,
            force_fresh = True,
            CHUNK_SIZE = 1024,
            SAVE_EVERY = 5,
            model = model,
            txt_data = reviews,
            N_CTX = 512,
            ind_path = "carp/examples/magiplots/embedding_inds.pt",
            enc_path = "carp/examples/magiplots/review_encs.pt",
            random_state = 69,
        )

        review_encs = torch.load("carp/examples/magiplots/review_encs.pt").float()
        inds = torch.load("carp/examples/magiplots/embedding_inds.pt")

        print("Performing UMAP...")
        umap_tform = UMAP(
            n_neighbors = 30,
            min_dist = 0.0,
            n_components = 2,
            metric = 'cosine',
            random_state = 42,
            low_memory = False
        ).fit(review_encs)

        joblib.dump(umap_tform, "carp/examples/magiplots/umap_tform.joblib")
        review_encs = umap_tform.transform(review_encs)

        print("Performing HDBSCAN Clustering...")
        hdbscan_clusterer = HDBSCAN(
            min_samples = 5,
            min_cluster_size = 30,
            prediction_data = True
        ).fit(review_encs)

        joblib.dump(hdbscan_clusterer, "carp/examples/magiplots/hdbscan_clusterer.joblib")
        labels = hdbscan_clusterer.labels_
        np.save("carp/examples/magiplots/cluster_labels.npy", labels)

    # Create the plot
    x = review_encs[:,0]
    y = review_encs[:,1]
    c = np.where(labels != -1, labels / labels.max(), labels) # normalize non error labels to [0, 1] for color
    
    cmap = plt.cm.RdYlGn
    c =  [(0, 0, 0, 1) if c_i == -1 else cmap(c_i) for c_i in c]

    plt.title("Reduced Embeddings And Clusters")

    plt.scatter(x, y, s = 2.5, c = c)

    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("carp/examples/magiplots/fig1.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    # ===== SENTIMENT ANALYSIS =====

    # Load the sentiment data if it was already computed
    try:
        sentiment_data = np.load("carp/examples/magiplots/sentiment_data.npy")
    except:
        # only extract sentiment for relevant reviews
        reviews = [reviews[i] for i in inds]
        reviews_keep = [reviews[i] for i in np.where(labels != -1)[0]]
        sentiment_data = extract_sentiment(reviews_keep)
        np.save("carp/examples/magiplots/sentiment_data.npy", sentiment_data)

    # Plot with a red to green colormap
    cmap = plt.cm.RdYlGn

    # Plot the sentiment data
    plt.title("Cluster Sentiments")

    plt.scatter(x, y, s = 2.5, c = sentiment_data, cmap = cmap)

    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("carp/examples/magiplots/fig2.pdf", format='pdf', bbox_inches='tight')
    plt.close()
