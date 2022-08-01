from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchtyping import TensorType


def scatter_with_names(
    x: np.ndarray,
    y: np.ndarray,
    names: Iterable[str],
    c: Iterable[float] = None,
    cmap: Callable = plt.cm.RdYlGn,
):
    """
    Creates an interactive scatterplot where user can hover over a point and
    see the corresponding string (i.e. review embeddings as points, reviews as the strings).
    Requires 2d points (x, y)

    :param x: x coordinates of points
    :type x: np.ndarray

    :param y: y coordinates of points
    :type y: np.ndarray

    :param names: list of strings to be displayed when hovering over points
    :type names: Iterable[str]

    :param c: list of floats between 0 and 1 to use as colors for the points, defaults to making all points black
    :type c: Iterable[float], optional

    :param cmap: Function to map values from [0, 1] to RGBA colors, defaults to plt cmap RdYlGn
    :type cmap: Callable, optional
    """
    names = np.array(names)
    norm = plt.Normalize(1, 4)
    fig, ax = plt.subplots()

    # Some preprocessing on c
    # -1's turned into black
    if c is None:
        c = [-1] * len(x)
    c = [(0, 0, 0, 1) if c_i == -1 else cmap(c_i) for c_i in c]

    sc = plt.scatter(x, y, s=2.5, c=c)

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    def update_annot(ind):
        index = ind["ind"][0]

        pos = sc.get_offsets()[index]
        annot.xy = pos
        text = names[index]
        annot.set_text(text)

        color = c[index]
        annot.get_bbox_patch().set_facecolor(color)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()


def spherical_coord(x: TensorType["batch", "d"]) -> TensorType["batch", "d - 1"]:
    """
    Converts points in R^d from euclidean coordinates to the angles parametrizing their locations on the unit d-sphere.
    Assumes the points are normalized (i.e. l2 norm of 1).

    :param x: normalized points in R^d
    :type x: TensorType["batch", "d"]

    :return: angles parametrizing the points on the unit d-sphere
    :rtype: TensorType["batch", "d - 1"]
    """
    n, d = x.shape
    # -> unit n-sphere points

    # phi[:,0:d-2] in [0, pi]
    # phi[:,d-2] in [-pi, pi]
    phi = torch.zeros_like(x)[:, :-1]  # -> (n, d-1)

    # value being curried to compute next angle
    curr = x[:, d - 1].pow(2) + x[:, d - 2].pow(2)

    # compute last angle first (it has weird formula conditioned on x_d)
    phi[:, d - 2] = torch.acos(x[:, d - 2] / curr.sqrt())
    phi[:, d - 2] = torch.where(
        x[:, d - 1] >= 0, phi[:, d - 2], 2 * np.pi - phi[:, d - 2]
    )

    # compute the rest
    for i in reversed(range(0, d - 2)):
        curr += x[:, i].pow(2)
        phi[:, i] = torch.acos(x[:, i] / curr.sqrt())

    return phi


def undo_spherical(phi):
    """
    Given angles parametrizing points on the unit d-sphere, returns the point in R^d.

    :param phi: angles parametrizing the points on the unit d-sphere
    :type phi: TensorType["batch", "d - 1"]

    :return: point in R^d
    :rtype: TensorType["batch", "d"]
    """
    n, d = phi.shape
    d += 1  # d angles makes d + 1 dim vector
    x = torch.ones(n, d, device=phi.device)

    # below forumlas for single [d] x, [d - 1] phi
    # x[0] = cos(phi[0])
    # x[i] = prod(sin(phi[0]), ..., sin(phi[i-1]) * cos(phi[0])
    # x[-1] = prod(sin(phi[0]), ..., sin(phi[-2]) * sin(phi[-1])

    x[:, : d - 1] *= torch.cos(phi)

    # make each phi[i] become product of sins of preceeding  elements
    for i in range(d - 1):
        crnt = torch.sin(phi[:, i])
        if i == 0:
            phi[:, i] = crnt
        else:
            phi[:, i] = crnt * phi[:, i - 1]

    x[:, 1:] *= phi

    return x
