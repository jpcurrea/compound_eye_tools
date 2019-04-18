# from matplotlib import pyplot as plt
# from matplotlib import colors
# from scipy import spatial
# import numpy as np
# import hdbscan
# from sty import fg

from compound_eye_tools import *

"""This reduces the filtered points to the filtered points in the eye only."""

# 0b. segment the eye using a density filter and HDBSCAN
head = np.load("./filtered_data.npy").astype(np.int16)
n = range(len(head))
selection = np.random.choice(n, int(np.round(max(n)/100.)))
extra = sorted(list(set(n) - set(selection)))
print("Training cluster algorithm on a random subset of the data.")
clusterer = hdbscan.HDBSCAN(min_cluster_size=50,
                            prediction_data=True,
                            algorithm='boruvka_kdtree').fit(head[selection])
counts = dict()
for lbl in sorted(set(clusterer.labels_)):
    counts[lbl] = sum(clusterer.labels_ == lbl)

# scatter plot of the selected data points before clustering
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 1000
w.show()
w.setWindowTitle('Moth Eye Ommatidia')

scatter = gl.GLScatterPlotItem(pos=head[selection], size=1)
w.addItem(scatter)

ret = input("HDBSCAN clusters a random subsection (1%) to get macro structures.")
w.removeItem(scatter)

txt_colors = ['red', 'green', 'yellow', 'blue', 'magenta']
pqt_colors = np.array([colors.to_rgba(c) for c in txt_colors])

scatters = []
clusters = dict()
cluster_centers = dict()

labels = sorted(set(clusterer.labels_))
labels = [lbl for lbl in labels if lbl >= 0]
labels = sorted(labels, key=lambda x: counts[x], reverse=True)
labels = labels[:min(len(txt_colors), len(labels))]
options = dict()

print("Top 5 clusters based on size (colors match the display):")
for num, (lbl, c, txt_color) in enumerate(zip(labels, pqt_colors, txt_colors)):
    if lbl >= 0:
        i = clusterer.labels_ == lbl
        group = head[selection][i]
        clusters[lbl] = group
        cluster_centers[lbl] = group.mean(0)
        scatter = gl.GLScatterPlotItem(pos=group, size=1, color=tuple(c))
        scatters += [scatter]
        # legend.addItem(scatter, name=lbl)
        w.addItem(scatter)
        options[num + 1] = lbl
        print(fg[txt_color] + "{}.\t{}".format(str(num + 1), len(group)) + fg.rs)


choice = int(input("Type the number corresponding to the eye cluster: "))
choice_lbl = options[choice]

for scatter in scatters:
    w.removeItem(scatter)

scatter = gl.GLScatterPlotItem(pos=clusters[choice_lbl], size=1)
w.addItem(scatter)

choice = input("Displaying sample eye data only. To find other points within cluster, press <Enter>.")
cluster_labels, strengths = hdbscan.approximate_predict(clusterer, head)
eye = head[cluster_labels == choice_lbl]

np.save("./eye_only.npy", eye)

w.removeItem(scatter)
scatter = gl.GLScatterPlotItem(pos=eye, size=1)
w.addItem(scatter)
choice = input("Crystalline cone data saved. Press <Enter> to close.")

# 1. convert to spherical coorinates by fitting a sphere with OLS
# 2. fit an n-degree polymonial, modelling each r as a function of theta and phi.
# 3. extract a range of points around the approximate surface
# 4. convert cone centers back to cartesian coordinates. For each center, find the nearest cluster of points within a generous radius .
# 5. Using our set of cone clusters, and the curvature of the thin sheet, we can take measurements relevant to the eye's optics.

