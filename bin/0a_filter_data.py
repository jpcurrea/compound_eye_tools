# from matplotlib import pyplot as plt
# from matplotlib import colors
# from scipy import spatial
# import numpy as np
# import hdbscan
# import PIL
# import sys
# import os


# from sty import fg

# def load_image(fn):
#     return np.asarray(PIL.Image.open(fn))

# def print_progress(part, whole):
#     prop = float(part)/float(whole)
#     sys.stdout.write('\r')
#     sys.stdout.write("[%-20s] %d%%" % ("="*int(20*prop), 100*prop))
#     sys.stdout.flush()

from compound_eye_tools import *

"""select the range of densities predominantly in the crystalline cones, filter the data, and save."""

# 0a. let user select, from images at different orientations,
# the range of densities corresponding to the crystalline cones
folder = "../cluster_data/21968/stack/"
fns = os.listdir(folder)
fns = [os.path.join(folder, fn) for fn in fns if fn.endswith(".tif")]

zs = len(fns)
imgs = []
print("Loading images:\n")
for num, fn in enumerate(fns):
    imgs += [load_image(fn)]
    print_progress(num, zs)
    
arr = np.array(imgs, dtype=np.uint16)
img = imgs[int(.5*zs)]

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg

app = QtGui.QApplication([])
slide = pg.image(arr)

ret = input("Use the bottom slide bar or the arrow keys to change slides and the right slide bar to select the range of density values corresponding predominantly to the crystaline cones. Left click and drag with the mouse to move the image around and use the scroll wheel to zoom in and out. Press <Enter> once you are happy with the filer.")

slide.close()

low, high = slide.getLevels()

xs = np.array([], dtype='uint16')
ys = np.array([], dtype='uint16')
zs = np.array([], dtype='uint16')

print("Extracting coordinate data: ")
for depth, img in enumerate(arr):
    y, x = np.where(
        np.logical_and(img <= high, img >= low))
    # y, x = np.where(image > 0)
    z = np.repeat(depth, len(x))
    xs = np.append(xs, x)
    ys = np.append(ys, y)
    zs = np.append(zs, z)
    print_progress(depth + 1, len(arr))

arr = np.array([xs, ys, zs], dtype=np.uint16).T
np.save("./filtered_data.npy", arr)
choice = input("Filtered data saved. Press <Enter> to close.")
