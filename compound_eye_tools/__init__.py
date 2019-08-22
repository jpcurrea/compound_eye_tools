from matplotlib import pyplot as plt
from matplotlib import colors
from scipy import spatial, interpolate
import numpy as np
import hdbscan
import PIL
import sys
import os
from sty import fg
import pandas as pd
import math
import pickle
from numpy import linalg as LA

from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog
from sklearn import cluster
import pyqtgraph.opengl as gl
import pyqtgraph as pg


def load_image(fn):
    """Import an image as a numpy array using the PIL."""
    return np.asarray(PIL.Image.open(fn))


def print_progress(part, whole, bar=True):
    """Print the part/whole progress bar."""
    prop = float(part)/float(whole)
    sys.stdout.write('\r')
    # sys.stdout.write("[%-20s] %d%%" % ("="*int(20*prop), round(100)*prop))
    if bar:
        prog_bar = '='*int(20*prop)
        st = f"[{prog_bar:20s}] {round(100 * prop)}%"
    else:
        st = f"{round(100*prop)}%"
    sys.stdout.write(st)
    sys.stdout.flush()


def fit_line(data, component=0):             # fit 3d line to 3d data
    """Use singular value decomposition (SVD) to find the best fitting vector to the data. 


    Keyword arguments:
    data -- input data points
    component -- the order of the axis used to decompose the data (default 0 => the first component vector which represents the plurality of the data
    """

    m = data.mean(0)
    max_val = np.round(2*abs(data - m).max()).astype(int)
    uu, dd, vv = np.linalg.svd(data - m)
    return vv[component]


def bootstrap_ci(arr, reps=1000, ci_range=[2.5, 97.5], stat_func=np.mean):
    """Use bootstrapping to generate a percentile range for a given statistic.


    Keyword arguments:
    arr -- input arr, preferably a numpy array
    reps -- the number of iterations for the bootstrap
    ci_range -- the percentile range to output
    stat_func -- the statistic to apply to the bootstrap (for instance, mean => 95% CI for the mean; std => 95% CI for the std; etc.)
    """
    pseudo_distro = np.random.choice(arr, (len(arr), reps))
    if stat_func is not None:
        pseudo_distro = stat_func(pseudo_distro, axis=1)
    l, h = np.percentile(pseudo_distro, ci_range, axis=0)
    return l, h


def rotate(arr, theta, axis=0):
    """Generate a rotation matrix and rotate input array along a single axis."""
    if axis == 0:
        rot_matrix = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]])
    elif axis == 1:
        rot_matrix = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 2:
        rot_matrix = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]])
    nx, ny, nz = np.dot(arr, rot_matrix).T
    nx = np.squeeze(nx)
    ny = np.squeeze(ny)
    nz = np.squeeze(nz)
    return np.array([nx, ny, nz])


def sphereFit(spX, spY, spZ):
    """Find best fitting sphere to x, y, and z coordinates using OLS."""
    f = np.zeros((len(spX), 1))
    f[:, 0] = (spX**2) + (spY**2) + (spZ**2)
    A = np.zeros((len(spX), 4))
    A[:, 0] = spX*2
    A[:, 1] = spY*2
    A[:, 2] = spZ*2
    A[:, 3] = 1
    C, residuals, rank, sigval = np.linalg.lstsq(A, f, rcond=None)
    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(t)
    return radius, np.squeeze(C[:-1]), residuals


def cartesian_to_spherical(pts, center=np.array([0, 0, 0])):
    """Convert rectangular/cartesian to spherical coordinates."""
    pts = pts - center
    radii = LA.norm(np.copy(pts), axis=1)
    theta = np.arccos(np.copy(pts)[:, 2]/radii)
    phi = np.arctan2(np.copy(pts)[:, 1], np.copy(pts)[:, 0]) + np.pi
    polar = np.array([theta, phi, radii])
    return polar.T


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class Points():
    """Stores coordinate data in both cartesian and spherical coordinates.


    Keyword arguments:
    center_points -- True => subtract out the center of mass (default True)
    polar -- if array of polar values are supplied, use them (default None)
    spherical_conversion -- True => use arr to calculate spherical coordinates (default True)
    rotate_com -- True => rotate points until center of mass is entirely located on the Z axis (default False)
    """

    def __init__(self, arr, center_points=True,
                 polar=None, spherical_conversion=True,
                 rotate_com=False):
        self.pts = np.array(arr)
        if arr.ndim > 1:
            assert self.pts.shape[1] == 3, ("Input array should have shape "
                                            "N x 3. Instead it has "
                                            "shape {} x {}.".format(
                                                self.pts.shape[0],
                                                self.pts.shape[1]))
        else:
            assert self.pts.shape[0] == 3, ("Input array should have shape "
                                            "3 or N x 3. Instead it has "
                                            "shape {}.".format(
                                                self.pts.shape))
            self.pts = self.pts.reshape((1, -1))
        self.shape = self.pts.shape
        self.center = np.array([0, 0, 0])
        self.polar = None
        if polar is not None:
            self.polar = polar
            self.theta, self.phi, self.radii = self.polar.T
        if spherical_conversion:
            # fit sphere
            x, y, z = self.pts.T
            self.radius, self.center, self.residuals = sphereFit(x, y, z)
            if center_points:
                # center points using the center of that sphere
                self.pts = self.pts - self.center
                self.center = self.center - self.center
            if rotate_com:
                # rotate points using the center of mass
                # 1. find center of mass
                com = self.pts.mean(0)
                # 2. rotate com along x axis (com[0]) until z (com[2]) = 0
                ang1 = np.arctan2(com[2], com[1])
                com1 = rotate(com, ang1, axis=0)
                rot1 = rotate(self.pts, ang1, axis=0).T
                # 3. rotate com along z axis (com[2]) until y (com[1]) = 0
                ang2 = np.arctan2(com1[1], com1[0])
                rot2 = rotate(rot1, ang2, axis=2).T
                self.pts = rot2
            # grab spherical coordinates of centered points
            self.spherical()
        self.x, self.y, self.z = self.pts.T

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        out = Points(self.pts[key], polar=self.polar[key], center_points=False)
        return out

    def spherical(self, center=None):
        if center is None:
            center = self.center
        self.polar = cartesian_to_spherical(self.pts, center=center)
        self.theta, self.phi, self.radii = self.polar.T

    def get_line(self):
        return fit_line(self.pts)


filetypes = [
    ('jpeg images', '*.jpeg *.jpg *.JPEG *.JPG'),
    ('png images', '*.png *.PNG'),
    ('tiff images', '*.tiff *.TIFF *.tff *.TFF *.tif *.TIF'),
    ('bmp images', '*.bmp *.BMP'),
    ('all types', '*.*')]

ftypes = [f"{fname} ({ftype})" for (fname, ftype) in filetypes]
ftypes = ";;".join(ftypes)


class fileSelector(QWidget):
    """Offer a file selection dialog box filtering common image filetypes."""

    def __init__(self, filetypes=ftypes):
        super().__init__()
        self.title = 'Select the images you want to process.'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.filetypes = filetypes
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.openFileNamesDialog()
        self.show()

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        self.files, self.ftype = QFileDialog.getOpenFileNames(
            self,
            "QFileDialog.getOpenFileNames()",
            "",
            self.filetypes,
            options=options)


class stackFilter():
    """Import image filenames filter images using upper and lower contrast bounds."""

    def __init__(self, fns=os.listdir("./"), save_coordinates=True, app=None,
                 save_fn="./filtered_data.npy"):
        """Import images using fns, a list of filenames."""
        self.save_fn = save_fn
        self.app = app
        self.fns = fns
        self.save = save_coordinates
        imgs = []
        print("Loading images:\n")
        for num, fn in enumerate(fns):
            try:
                imgs += [load_image(fn)]
            except:
                print(f"{fn} failed to load.")
            print_progress(num, len(fns))

        assert len(imgs) > 0, "All images failed to load."
        self.imgs = np.array(imgs, dtype=np.uint16)

    def contrast_filter(self):
        """Use pyqtgraph's image UI to select lower an upper bound contrasts."""
        # if there is no defined application instance, make one
        if self.app is None:
            self.app = QApplication.instance()
            if self.app is None:
                self.app = QApplication([])

        self.image_UI = pg.image(self.imgs)  # use the image UI from pyqtgraph
        self.image_UI.setPredefinedGradient('greyclip')
        self.app.exec_()      # allow the application to run until closed
        # grab low and high bounds from UI
        self.low, self.high = self.image_UI.getLevels()
        xs = np.array([], dtype='uint16')
        ys = np.array([], dtype='uint16')
        zs = np.array([], dtype='uint16')

        print("Extracting coordinate data: ")
        for depth, img in enumerate(self.imgs):
            y, x = np.where(
                np.logical_and(img <= self.high, img >= self.low))
            # y, x = np.where(image > 0)
            z = np.repeat(depth, len(x))
            xs = np.append(xs, x)
            ys = np.append(ys, y)
            zs = np.append(zs, z)
            print_progress(depth + 1, len(self.imgs))
            print("./")

        self.arr = np.array([xs, ys, zs], dtype=np.uint16).T
        if self.save:
            print(f"Saving coordinates to {self.save_fn}")
            np.save(self.save_fn, self.arr)


class ScatterPlot3d():
    """Plot 3d datapoints using pyqtgraph's GLScatterPlotItem."""

    def __init__(self, arr, color=(1, 1, 1, 1), size=1, app=None, window=None):
        self.arr = arr
        self.color = color
        self.size = size
        self.app = app
        self.window = window

        self.n, self.dim = self.arr.shape
        assert self.dim == 3, ("Input array should have shape "
                               "N x 3. Instead it has "
                               "shape {} x {}.".format(
                                   self.n,
                                   self.dim))

    def show(self):
        if self.app is None:
            self.app = QApplication.instance()
            if self.app is None:
                self.app = QApplication([])
        if self.window is None:
            self.window = gl.GLViewWidget()
            self.window.setWindowTitle("3D Scatter Plot")
        self.scatter_GUI = gl.GLScatterPlotItem(
            pos=self.arr, size=self.size, color=self.color)
        self.window.addItem(self.scatter_GUI)
        self.window.show()
        self.app.exec_()


class ScatterPlot2d():
    """Plot 3d datapoints using pyqtgraph's GLScatterPlotItem."""

    # def __init__(self, arr, color=(1, 1, 1, 1), size=[1], app=None, window=None):
    def __init__(self, arr, color=[1], app=None, window=None, axis=None):
        self.arr = np.array(arr)
        self.color = np.array(color)
        # self.size = np.array(size)
        self.app = app
        self.window = window
        self.axis = axis

        self.n, self.dim = self.arr.shape
        assert self.dim == 2, ("Input array should have shape "
                               "N x 2. Instead it has "
                               "shape {} x {}.".format(
                                   self.n,
                                   self.dim))
        # assert len(self.color) in [self.n, 1], (
        #     "Input color array should have length 1 or N. Instead it has "
        #     "shape {}.".format(self.color.shape))
        if len(self.color) == 1:
            color = pg.intColor(self.color[0])
            # color = [[self.color
            self.color = np.repeat(color, self.n)
        if len(self.color) == 4:
            self.color = np.repeat([self.color], self.n)
        # if len(self.size) != self.n:
        #     self.color = np.repeat(self.size, self.n)

    def show(self):
        if self.app is None:
            self.app = QApplication.instance()
            if self.app is None:
                self.app = QApplication([])
        if self.window is None:
            self.window = pg.GraphicsLayoutWidget()
            self.window.setWindowTitle("2D Scatter Plot")
        if self.axis is None:
            self.axis = self.window.addPlot()
        self.scatter_GUI = pg.ScatterPlotItem(
            # pos=self.arr, size=self.size, color=self.color)
            pos=self.arr, color=self.color)
        spots = []
        # for pos, size, color in zip(self.arr, self.size, self.color):
        for pos, color in zip(self.arr, self.color):
            spots.append({
                'pos': pos,
                # 'size': size,
                'pen': {'color': 'w'},
                'brush': color})
        self.scatter_GUI.addPoints(spots)
        self.axis.addItem(self.scatter_GUI)
        self.window.show()
        self.app.exec_()


def main():
    app = QApplication([])
    # file_UI = fileSelector()
    # file_UI.close()
    # fns = file_UI.files

    folder = "/home/pbl/Desktop/programs/ommatidia_counter/bombus/morphosourceMedia_03_13_19_164700/LU_3_14_AM_F_5_M35381-65646_Apis_mellifera_Scan_60185/60185/test_bee_eye/cleaned_up/"
    os.chdir(folder)
    # testing:
    # fns = os.listdir("./")
    # fns = sorted([os.path.join(folder, fn)
    #               for fn in fns if fn.endswith(".tif")])

    # SF = stackFilter(fns, app=app)
    # SF.contrast_filter()
    # arr = SF.arr

    arr = np.load("./filtered_data.npy")  # testing
    arr = arr - arr.mean(0)
    arr = Points(arr, center_points=True, rotate_com=True)
    # scatter = ScatterPlot3d(arr.pts)          # testing
    # scatter.show()

    # polar_vals = arr.polar
    # polar_vals[:, 2] /= 100
    # polar = ScatterPlot3d(arr.polar)          # testing
    # polar.show()

    scatter = ScatterPlot2d(arr.polar[:, :2])
    scatter.show()

    return arr


scatter = main()

if __name__ == "__main__":
    main()
