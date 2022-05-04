import torch
import numpy as np

from umap import UMAP
import joblib

from carp.examples.visualization.plot_util import scatter_with_names
from carp.examples.encodings.encode_reviews import enc_reviews

from carp.configs import CARPConfig
from carp.pytorch.data import *
from carp.pytorch.model.architectures.carp import CARP

# This script performs a UMAP reduction on review embeddings of randomly sampled reviews
# from the Story-Critique dataset. Any other dataset should be trivial to plug in.
# The reduced embeddings are then plotted such that hovering over a single embedding (a point in R^2),
# shows the review (or more generally, text sample) it came from.
if __name__ == "__main__":
    # Load dataset first
    print("Load Dataset...")
    pipeline = BaseDataPipeline(path = "carp/dataset")

    # First try loading all the data in case computations were already performed

    try:
        umap_tform = joblib.load("carp/examples/visualization/umap_tform.joblib")
        review_encs = torch.load("carp/examples/visualization/review_encs.pt")
        review_inds = torch.load("carp/examples/visualization/review_inds.pt")
    except:
        print("Could not find previous data, computing now...")
        
        # Load model
        print("Load Model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = CARPConfig.load_yaml("configs/carp_l.yml")
        model = CARP(config.model)
        model.load("checkpoints/CARP_L/")
        model = model.to(device)

        # Generate encodings
        print("Generate Encodings...")
        enc_reviews(
            N_SAMPLES = 10000, force_fresh = False, CHUNK_SIZE = 256, SAVE_EVERY = 5,
            model = model, txt_data = pipeline.reviews,
            ind_path = "carp/examples/visualization/review_inds.pt",
            enc_path = "carp/examples/visualization/review_encs.pt",
            random_state = 42
        )
        review_encs = torch.load("carp/examples/visualization/review_encs.pt")
        review_inds = torch.load("carp/examples/visualization/review_inds.pt")

        # Perform UMAP
        print("Fit UMAP...")
        tform = UMAP(
            n_neighbors = 30,
            min_dist = 0.0,
            n_components = 2,
            metric = 'cosine',
            random_state = 42,
            low_memory = False
        ).fit(review_encs)
    
        # Save UMAP transformation
        joblib.dump(tform, "carp/examples/visualization/umap_tform.joblib")
    
    # get selected reviews
    review_txt = [pipeline.reviews[i] for i in review_inds]
    review_encs = review_encs.cpu().numpy()
    review_encs = tform.transform(review_encs)

    x = review_encs[:,0]
    y = review_encs[:,1]

    # Plot
    scatter_with_names(x, y, review_txt)


