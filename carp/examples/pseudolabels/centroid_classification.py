import torch
import torch.nn.functional as F
import numpy as np
from umap import UMAP

from torchtyping import TensorType

import joblib
import sys

from carp.configs import CARPConfig
from carp.pytorch.data import *
from carp.pytorch.model.architectures import BaseModel
from carp.pytorch.model.architectures.carp import CARP


def generate_centroids(
        review_encs : TensorType["n", "d"], labels : np.ndarray, umap_tform : UMAP = None
    ) -> TensorType["m", "d"]:
    """
    Given reduction transformation, review encodings, and labels, generate centroids per label in reduced space

    :param review_encs: Review embeddings, [n, d] tensor
    :type review_encs: torch.tensor

    :param labels: Labels for review embeddings, [n] array
    :type labels: np.ndarray

    :param umap_tform: UMAP transformation to use for reduction
    :type umap_tform: UMAP, optional

    :return: Centroids in reduced space, normalized, [m, d] tensor, where m is number of labels and d is either dim of review_encs or 2 if umap_tform is given
    :rtype: torch.tensor
    """
    if umap_tform is not None: 
        review_encs = umap_tform.transform(review_encs)
        review_encs = torch.from_numpy(review_encs)
        means = torch.zeros(labels.max(), 2)
    else:
        means = torch.zeros(labels.max() + 1, review_encs.shape[1]) 

    for i in range(labels.max() + 1):
        inds = np.where(labels == i)[0]

        means[i] = review_encs[inds].mean(0)
    
    if umap_tform is None:
        means = F.normalize(means) # should be points on d-sphere if still in original space
    
    return means

def classify_encoding(encoding : TensorType["d"], centroids : TensorType["m", "d"], metric : str = "cosine") -> int:
    """
    Given encoding and centroids, classify encoding as belonging to one of the clusters the centroids correspond to

    :param encoding: Single review encoding, [n] tensor
    :type encoding: torch.tensor

    :param centroids: Centroids to act as classifiers, [m, d] tensor where m is number of classifiers/labels
    :type centroids: torch.tensor

    :param metric: Metric by which to compare encoding to centroids. Can either be "cosine" or "euclidean"
    :type metric: str

    :return: classification label (i.e. pseudolabel)
    :rtype: int
    """
    encoding = encoding.float()
    encoding /= encoding.norm()

    if metric == "cosine":
        sims = centroids @ encoding # -> [m]
        # this is sufficient for cos sim if both are normalized
    elif metric == "euclidean":
        sqr_dist = encoding[None,:] - centroids # -> [m, d]
        sqr_dist = sqr_dist ** 2
        sqr_dist = sqr_dist.sum(1) # -> [m]
        sims = -1 * sqr_dist # max similarity is minimum sqr distance
    else:
        print("Warning: Invalid metric")
        return -1
    
    return sims.argmax().item()

if __name__ == "__main__":
    use_umap = False
    # Decide whether or not to use UMAP
    if len(sys.argv) >= 2:
        if sys.argv[1] == "UMAP" or sys.argv[1] == "umap":
            use_umap = True

    try:
        review_encs = torch.load("carp/examples/pseudolabels/review_encs.pt")
        umap_tform = joblib.load("carp/examples/pseudolabels/umap_tform.joblib")
        labels = np.load("carp/examples/pseudolabels/cluster_labels.npy")
    except:
        print("Could not find nessecary objects. Please ensure you have run umap_clustering.py")
        exit()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use CARP Large by default
    print("Load Model...")
    config = CARPConfig.load_yaml("configs/carp_l.yml")
    model = CARP(config.model)
    model.load("checkpoints/CARP_L/")
    model = model.to(device)

    # encode a single text sample with model
    def encode(txt, mode): # mode as R or P
        if mode == "P":
            tokenize = model.passage_encoder.call_tokenizer
        elif mode == "R":
            tokenize = model.review_encoder.call_tokenizer

        txt = [txt]
        tok_out = tokenize(txt)

        x = tok_out["input_ids"].to(device)
        mask = tok_out["attention_mask"].to(device)
        enc_input = BatchElement(x, mask)

        with torch.no_grad():
            if mode == "P":
                encs = model.encode_passages(enc_input).hidden
            elif mode == "R":
                encs = model.encode_reviews(enc_input).hidden
        enc = encs.cpu().squeeze()
        return enc

    print("Getting Centroids...")
    centroids = generate_centroids(review_encs, labels, umap_tform = umap_tform if use_umap else None)

    print("Enter P or R for type of text to classify followed by the text, i.e.")
    print("P The quick brown fox jumped over the lazy dog")
    print("or")
    print("R There's too many adjectives here")
    print("type exit to exit")

    while True:
        try:
            txt = input()
            if txt == "exit": break

            # split txt into mode and actual string, then encode and classify
            mode, txt = txt[0], txt[2:]
            enc = encode(txt, mode)
            label = classify_encoding(enc, centroids, "euclidean" if use_umap else "cosine")

            print("Predicted Label: {}".format(label))

        except KeyboardInterrupt:
            exit()
        except:
            print("Invalid Input")

