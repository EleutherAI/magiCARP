from cgi import test
from torch import nn
from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader

from typing import Callable

# This evaluation script has a predefined selection of classification tasks
# To evaluate a representation learning model, it loads a classification dataset
# for that modality. A single linear head is trained as a classifier to classify
# the model embeddings. Test accuracy is then returned.

# This is largely TODO as one would have to define their own dataset to use here
# Reccomended to use a very small dataset (~5k) as this is for evaluation entirely
task_datasets = {
    "default" : None
}

task_output  = {
    "default" : 4
}

def lh_classifier_eval(embedder : nn.Module, preprocess : Callable, task : str = "default",
    emb_chunk_size : int = 256, clf_chunk_size : int = -1, clf_epochs : int = 10, test_frac : float = 0.5
    , device = 'cpu'):
    """
    Evaluates an embedding model by using its embeddings for a classification task. Returns the test accuracy of
    the classifier trained on its embeddings.

    :param embedder: Encoder for modality
    :type embedder: torch.nn.Module

    :param preprocess: Function to preprocess data for the embedder
    :type preprocess: Callable
    
    :param task: The task to classify
    :type task: str

    :param chunk_size: Chunk size for embedding model to process dataset
    :type chunk_size: int

    :param clf_chunk_size: Chunk size for classifier to process embeddings. If -1, uses entire dataset.
    :type clf_chunk_size: int

    :param clf_epochs: Number of epochs to train the classifier for
    :type clf_epochs: int

    :param test_frac: Proportion of dataset embeddings to use as test set for classifier, should be in (0, 1)
    :type test_frac: float

    :return: Returns accuracy of classifier on test set, in (0, 1)
    :rtype: float
    """
    ds = task_datasets["default"]
    out_dim = task_output["default"]

    clf = nn.Linear(embedder.latent_dim, out_dim)
    clf.to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr = 1e-3)

    loss_fn = nn.NLLLoss()

    loader = DataLoader(ds, batch_size = emb_chunk_size)

    enc_batches, label_batches = [], []

    with torch.no_grad():
        for batch in loader:
            sample, label = batch
            label_batches.append(label)

            enc = embedder(preprocess(sample))
            enc = F.normalize(enc)
            enc_batches.append(enc)
    
    encs = torch.cat(enc_batches).to(device)
    labels = torch.cat(label_batches).to(device)

    test_ind = int(len(encs) * test_frac)

    train_enc = encs[:test_ind]
    test_enc = encs[test_ind:]
    train_labels = labels[:test_ind]
    test_labels = labels[test_ind:]

    def batch(size, enc, label):
        if size == -1:
            enc_batches = [enc]
            label_batches = [label]
        else:
            enc_batches = enc.chunk(len(enc) // size)
            label_batches = label.chunk(len(labels) // size)

        return enc_batches, label_batches

    enc_batches, label_batches = batch(clf_chunk_size, train_enc, train_labels)

    # Train classifier
    for epoch in range(clf_epochs):
        for (batch, label) in zip(enc_batches, label_batches):
            opt.zero_grad()
            logits = clf(batch)
            loss = loss_fn(logits, label)
            loss.backward()
            opt.step()
    
    enc_batches, label_batches = batch(clf_chunk_size, test_enc, test_labels)

    # Get  test accuracy
    correct = 0 
    total = len(test_labels)
    with torch.no_grad():
        for (batch, label) in zip(enc_batches, label_batches):
            label_pred = clf(batch).argmax(-1)
            correct += ((label_pred == label).sum())
    
    return correct/total






    