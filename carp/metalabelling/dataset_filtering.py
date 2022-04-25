import torch
import numpy as np
import matplotlib.pyplot as plt


def filter_by_thresholds(sims, threshold_1, threshold_2):
    """
    Return indices of rows where the maximum value is greater than threshold_1,
    and the largest value is greater than the second largest value by at least threshold_2
    Meant to remove samples that don't have a sufficiently high similarity with any centroid,
    or are ambiguous due to high proximity to multiple centroids

    :param sims: N x K tensor of similarities between N samples and K centroids
    :type sims: tensor

    :param threshold_1: What the highest similarity must be greater than to pass filter
    :type threshold_1: int

    :param threshold_2: What the difference between highest and second highest similarities must be greater than to pass filter
    :type threshold_2: int

    :rtype: tensor
    :return: tensor of indices of rows passing filter
    """

    a = threshold_1
    b = threshold_2

    # Sort the similarities in descending order
    sorted_sims, _ = torch.sort(sims, dim=1, descending=True)

    filter_ = (sorted_sims[:, 0] > a) & (sorted_sims[:, 0] - sorted_sims[:, 1] > b)

    inds = filter_.nonzero().squeeze()

    return inds

if __name__ == "__main__":
    sims = torch.load("metalabel_data/dataset_centroid_dists_overall.pt").float()

    print("Number of samples before filtering: {}".format(sims.shape[0]))

    sorted_sims = torch.sort(sims, dim = 1, descending = True)[0]

    def hist(x):
        plt.bar(np.arange(len(x)), x, 1)

    # Plot of distribution of similarity scores averaged over all rows
    # Shows what max and minimum sims were on average
    hist(sorted_sims.mean(0).numpy())
    plt.show()
    plt.close()

    # Now want a distribution of just the max values
    plt.hist(sorted_sims[:,0].numpy())
    plt.show()
    plt.close()

    # Filter based on just maximum value
    inds = filter_by_thresholds(sims, 0.5, 0.0)
    sorted_sims_fil_1 = sorted_sims[inds]

    # Want to see distribution of max value - second max value
    deltas = sorted_sims_fil_1[:,0] - sorted_sims_fil_1[:,1]
    plt.hist(deltas.numpy(), bins = 100)
    plt.show()
    plt.close()

    # 0.4 seems like a good cut off
    # Filter based on just maximum value
    inds = filter_by_thresholds(sims, 0.5, 0.4)
    sims = sims[inds]
    sorted_sims = sorted_sims[inds]
    print("Number of samples after filtering: {}".format(sims.shape[0]))

    # Let's redo some of the earlier plots
    hist(sorted_sims.mean(0).numpy())
    plt.show()
    plt.close()

    plt.hist(sorted_sims[:,0].numpy())
    plt.show()
    plt.close()