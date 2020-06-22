import matplotlib.pyplot as plt
from matplotlib import rcParams

# path to output figures
path = "/tmp/"

M = 1

# layout
dpi = 300  # resolution of saved figures
linewidth = 3 * M
legend_location = "lower right"
legend_bbox_to_anchor = (0.99, 0.06)  # offset of lower right legend corner
legend_fontsize = 20 * M
legend_alpha = 1
tick_fontsize = 20 * M
label_fontsize = 25 * M
label_padding = 8 * M
title_fontsize = 25 * M
rcParams["font.family"] = "Times New Roman"
format = "png"
markersize = 10 * M
markeredgewidth = 2 * M


def savefig(name, absolute_path=False):
    fig = plt.gcf()
    if not absolute_path:
        f = "{}{}.{}".format(path, name, format)
    else:
        f = name + "." + format
    fig.savefig(f, format=format, dpi=dpi)
    print("Saved figure to {}".format(f))


plt.xlim([2015, 2400])
