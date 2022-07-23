import torch
import torch.nn.functional as F

import numpy as np
import random

import wandb

from hdbscan import HDBSCAN
from umap import UMAP

from typing import Callable, Iterable

def umap_wandb(embedder : torch.nn.Module, data : Iterable[any], preprocess : Callable, size : int = 1000, chunk_size : int = 512):
    """
    Creates a wandb scatterplot of reduced embeddings on data from provided embedder.

    :param embedder: The embedding model to use
    :type embedder: torch.nn.Module

    :param data: The data to use. Can be anything.
    :type data: Iterable[any]

    :param preprocess: The preprocessing function to apply to data before it goes into embedder.
    :type preprocess: Callable

    :param size: The number of embeddings to produce for the plot. Small values should still reveal a lot about representation space.
    :type size: int

    :param chunk_size: Size of chunks to embed with embedder
    :type chunk_size: int

    :return: wandb scatterplot of the UMAP'd embeddings
    :rtype: wandb.plot.scatter
    """

    torch.manual_seed(0)
    inds = torch.randperm(len(data))[:size]

    if type(data) == list:
        data = [data[i] for i in inds]
    else:
        data = data[inds]

    def chunk(l, s):
        return [l[i:i + s] for i in range(0, len(l), s)]

    data = chunk(data, chunk_size)
    embs = []

    with torch.no_grad():
        for data_chunk in data:
            enc = embedder(preprocess(data_chunk))
            enc = F.normalize(enc)
            embs.append(enc)

    embs = torch.cat(embs).to('cpu')

    z = UMAP(
            n_neighbors = 30,
            min_dist = 0.0,
            n_components = 2,
            metric = 'cosine',
            random_state = 42,
            low_memory = False
    ).fit_transform(embs)

    x = embs[:,0]
    y = embs[:,1]

    return wandb.plot.scatter(embs, "x", "y", "UMAP Embeddings")






