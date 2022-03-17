import matplotlib.pyplot as plt
import numpy as np
import torch


# Parameterize x as angles of a unit hypersphere
# Further explanation in notebook
# assume x is normalized (l2 norm x_i = 1)
# assumes d >= 3
def spherical_coord(x):
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


# scatter plot with labels/names
# x = array like
# y = array like same length as x
# names = list of strings
# source: https://stackoverflow.com/q/7908636
def scatter_with_names(x, y, names):
    c = np.random.randint(1, 5, size=len(x))
    names = np.array(names)
    norm = plt.Normalize(1, 4)
    cmap = plt.cm.RdYlGn
    fig, ax = plt.subplots()
    sc = plt.scatter(x, y, cmap=cmap, norm=norm)

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

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}, {}".format(
            " ".join(list(map(str, ind["ind"]))),
            " ".join([names[n] for n in ind["ind"]]),
        )
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
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
