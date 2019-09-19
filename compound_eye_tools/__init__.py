import pdb
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy import spatial, interpolate, ndimage
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

import fly_eye as fe

blue, green, yellow, orange, red, purple = [(0.30, 0.45, 0.69), (0.33, 0.66, 0.41), (
    0.83, 0.74, 0.37), (0.78, 0.50, 0.16), (0.77, 0.31, 0.32), (0.44, 0.22, 0.78)]


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


def angle_between_skew_vectors(
        position_vector_1, direction_vector_1,
        position_vector_2, direction_vector_2):
    """uses derivation from 
    https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points
    """
    p1, d1 = position_vector_1, direction_vector_1
    p2, d2 = position_vector_2, direction_vector_2
    n = np.cross(d1, d2)
    n1, n2 = np.cross(d1, n), np.cross(d2, n)
    c1 = p1 + np.dot((np.dot((p2 - p1), n2)/(np.dot(d1, n2))), d1)
    c2 = p2 + np.dot((np.dot((p1 - p2), n1)/(np.dot(d2, n1))), d2)
    midpoint = np.mean([c1, c2], axis=0)
    centered_p1, centered_p2 = p1 - midpoint, p2 - midpoint
    c1, c2 = centered_p1, centered_p2
    theta = np.arccos(
        np.dot(c1, c2)/(LA.norm(c1) * LA.norm(c2)))
    return theta


def load_Points(fn):
    with open(fn, "rb") as pickle_file:
        out = pickle.load(pickle_file)
    return out


class Points():
    """Stores coordinate data in both cartesian and spherical coordinates.


    Keyword arguments:
    center_points -- True => subtract out the center of mass (default True)
    polar -- if array of polar values are supplied, use them (default None)
    spherical_conversion -- True => use arr to calculate spherical coordinates (default True)
    rotate_com -- True => rotate points until center of mass is entirely located on the Z axis
    (default False)
    """

    def __init__(self, arr, center_points=True,
                 polar=None, spherical_conversion=True,
                 rotate_com=True):
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
        self.original_pts = self.pts
        self.shape = self.pts.shape
        self.center = np.array([0, 0, 0])
        self.polar = None
        if polar is not None:
            self.polar = polar
            self.theta, self.phi, self.radii = self.polar.T
        if spherical_conversion:
            # fit sphere
            x, y, z = self.pts.T
            self.radius, self.center, self.resid_ = sphereFit(x, y, z)
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
        out = Points(self.pts[key], polar=self.polar[key])
        return out

    def spherical(self, center=None):
        if center is None:
            center = self.center
        self.polar = cartesian_to_spherical(self.pts, center=center)
        self.theta, self.phi, self.radii = self.polar.T
        self.residuals = self.radii - self.radius

    def get_line(self):
        return fit_line(self.pts)

    def rasterize(self, mode='polar', axes=[0, 1], pixel_length=.01):
        """Rasterize float coordinates in to a grid defined by min and max vals
        and sampled at pixel_length.
        """
        if mode == 'pts':
            arr = self.pts
        if mode == 'polar':
            arr = self.polar
        x, y, z = arr.T

        xs = np.arange(x.min(), x.max(), pixel_length)
        ys = np.arange(y.min(), y.max(), pixel_length)

        avg = []
        for col_num, (x1, x2) in enumerate(zip(xs[:-1], xs[1:])):
            col = []
            in_column = np.logical_and(x >= x1, x < x2)
            in_column = arr[in_column]
            for row_num, (y1, y2) in enumerate(zip(ys[:-1], ys[1:])):
                in_row = np.logical_and(
                    in_column[:, 1] >= y1, in_column[:, 1] < y2)
                avg += [in_row.sum()]
            print_progress(col_num, len(xs) - 1)
        avg = np.array(avg)
        avg = avg.reshape((len(xs) - 1, len(ys) - 1))
        self.raster = avg
        xs = xs[:-1] + (pixel_length / 2.)
        ys = ys[:-1] + (pixel_length / 2.)
        return self.raster, (xs, ys)

    def fit_surface(self, mode='polar', outcome_axis=0, pixel_length=.01):
        """Find cubic interpolation surface of one axis using the other two."""
        if mode == 'pts':
            arr = self.pts
        if mode == 'polar':
            arr = self.polar
        x, y, z = arr.T

        # reduce data using a 2D rolling average
        xs = np.arange(x.min(), x.max(), pixel_length)
        ys = np.arange(y.min(), y.max(), pixel_length)
        avg = []
        for col_num, (x1, x2) in enumerate(zip(xs[:-1], xs[1:])):
            col = []
            in_column = np.logical_and(x >= x1, x < x2)
            in_column = arr[in_column]
            for row_num, (y1, y2) in enumerate(zip(ys[:-1], ys[1:])):
                in_row = np.logical_and(
                    in_column[:, 1] >= y1, in_column[:, 1] < y2)
                if any(in_row):
                    avg += [np.mean(in_column[in_row], axis=0)]
            print_progress(col_num, len(xs) - 1)
        avg = np.array(avg)

        # filter outlier points by using bootstraped 95% confidence band (not of the mean)
        low, high = np.percentile(avg[:, 2], [.5, 99.5])
        avg = avg[np.logical_and(avg[:, 2] >= low, avg[:, 2] < high)]
        avg_x, avg_y, avg_z = avg.T

        # interpolate through avg to get 'cross section' using
        # bivariate spline (bisplrep)
        tck = interpolate.bisplrep(avg_x, avg_y, avg_z)
        z_new = []
        for xx, yy in zip(x, y):
            z_new += [interpolate.bisplev(xx, yy, tck)]
        self.surface = np.array(z_new)

    def get_polar_cross_section(self, thickness=.1, pixel_length=.01):
        """Find best fitting surface of radii using phis and thetas."""
        self.fit_surface(mode='polar', pixel_length=pixel_length)
        # find distance of datapoints from surface (ie. residuals)
        self.residuals = self.radii - self.surface
        # choose points within 'thickness' proportion of residuals
        self.cross_section_thickness = np.percentile(
            abs(self.residuals), thickness * 100)
        self.surface_lower_bound = self.surface - self.cross_section_thickness
        self.surface_upper_bound = self.surface + self.cross_section_thickness
        cross_section_inds = np.logical_and(
            self.radii <= self.surface_upper_bound,
            self.radii > self.surface_lower_bound)
        self.cross_section = self[cross_section_inds]

    def save(self, fn):
        """Save using pickle."""
        with open(fn, "wb") as pickle_file:
            pickle.dump(self, pickle_file)


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

    def __init__(self, fns=os.listdir("./"), app=None):
        """Import images using fns, a list of filenames."""
        self.app = app
        self.fns = fns
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

        self.arr = np.array([xs, ys, zs], dtype=np.uint16).T


class ScatterPlot3d():
    """Plot 3d datapoints using pyqtgraph's GLScatterPlotItem."""

    def __init__(self, arr, color=None, size=1, app=None, window=None,
                 colorvals=None, cmap=plt.cm.viridis):
        self.arr = arr
        self.color = color
        self.cmap = cmap
        self.size = size
        self.app = app
        self.window = window

        self.n, self.dim = self.arr.shape
        assert self.dim == 3, ("Input array should have shape "
                               "N x 3. Instead it has "
                               "shape {} x {}.".format(
                                   self.n,
                                   self.dim))
        if isinstance(self.color, (tuple, list)):
            self.color = np.array(self.color)
        elif self.color is None and colorvals is None:
            colorvals = np.ones(self.arr.shape[0])
        elif self.color is None and colorvals is not None:
            colorvals = (colorvals - colorvals.min()) / \
                (colorvals.max() - colorvals.min())
            self.color = np.array([self.cmap(c) for c in colorvals])
        if self.color.max() > 1:
            self.color = self.color / self.color.max()

        self.plot()

    def plot(self):
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

    def show(self):
        self.window.show()
        self.app.exec_()


class ScatterPlot2d():
    """Plot 2d datapoints using pyqtgraph's GLScatterPlotItem."""

    # def __init__(self, arr, color=(1, 1, 1, 1), size=[1], app=None, window=None):
    def __init__(self, arr, color=None, app=None, window=None, size=1,
                 axis=None, colorvals=None, cmap=plt.cm.viridis, scatter_GUI=None):
        self.arr = np.array(arr)
        self.color = color
        self.colorvals = colorvals
        self.cmap = cmap
        self.size = size
        self.app = app
        self.window = window
        self.axis = axis
        self.scatter_GUI = scatter_GUI

        self.n, self.dim = self.arr.shape
        assert self.dim == 2, ("Input array should have shape "
                               "N x 2. Instead it has "
                               "shape {} x {}.".format(
                                   self.n,
                                   self.dim))
        # how to handle color array:
        # if float or int, repeat float/int and convert to pg.intColor

        if isinstance(self.color, (tuple, list)):
            self.color = np.array(self.color)
            assert self.color.shape[0] in [
                4, self.n], "input colors are of the wrong length/shape"
        elif self.color is None and colorvals is None:
            colorvals = np.ones(self.arr.shape[0])
        elif self.color is None and colorvals is not None:
            assert len(
                colorvals) == self.n, "inpur colorvals are of the wrong length"
            colorvals = (colorvals - colorvals.min()) / \
                (colorvals.max() - colorvals.min())
            self.color = np.array([self.cmap(c) for c in colorvals])
        if len(self.color) == 4:
            self.color = np.tile(color, self.n).reshape(self.n, 4)
        # if self.color.max() > 1:
        self.color = self.color / self.color.max()
        self.color = np.round(255 * self.color).astype(int)
        # if self.colorvals is None and self.color is None:
        #     self.color = np.repeat(pg.intColor(1), len(self.arr))

        # if self.color is None and self.colorvals is not None:
        #     assert len(self.colorvals) == len(self.arr), (
        #         "Colorvals array is of the wrong length."
        #         f" Array should have length {len(self.arr)},"
        #         f" but is {len(self.colorvals)}.")
        #     if max(self.colorvals) > 255 or max(self.colorvals) <= 1:
        #         self.colorvals = (self.colorvals - self.colorvals.min()) / \
        #             (self.colorvals.max() - self.colorvals.min())
        #         self.color = 255 * np.array([self.cmap(c)
        #                                      for c in self.colorvals])

        # elif isinstance(self.color, (float, int)):
        #     self.color = np.repeat(self.color, len(self.arr))
        #     self.color = np.array(self.color)

        # elif isinstance(self.color, (list, tuple)):
        #     assert len(self.color) in [len(self.arr), 1], (
        #         "Color array is of the wrong length."
        #         f" Array should have length of 1 or {len(self.arr)},"
        #         f" but is {len(self.color)}.")
        #     self.color = np.array(self.color)

        # elif isinstance(self.color, (list, tuple)):
        #     assert len(self.color) in [len(self.arr), 1], (
        #         "Color array is of the wrong length."
        #         f" Array should have length of 1 or {len(self.arr)},"
        #         f" but is {len(self.color)}.")
        #     self.color = np.array(self.color)

        # if self.color.ndim > 1:
        #     if self.color.shape[1] != 4:
        #         color = []
        #         for c in self.color:
        #             color += [pg.intColor(c[0])]
        #         self.color = np.array(color)
        # elif len(self.color) == 4:
        #     self.color = np.repeat([self.color], self.n)

        if isinstance(self.size, (float, int)):
            self.size = np.repeat(self.size, self.n)

        self.plot()

    def plot(self):
        if self.app is None:
            self.app = QApplication.instance()
            if self.app is None:
                self.app = QApplication([])
        if self.window is None:
            self.window = pg.GraphicsLayoutWidget()
            self.window.setWindowTitle("2D Scatter Plot")
        if self.axis is None:
            self.axis = self.window.addPlot()
        if self.scatter_GUI is None:
            self.scatter_GUI = pg.ScatterPlotItem()
        # pos=self.arr, size=self.size, color=self.color)
        # pos=self.arr, color=)
        spots = []
        # for pos, size, color in zip(self.arr, self.size, self.color):
        for pos, color, size in zip(self.arr, self.color, self.size):
            if isinstance(color, np.ndarray):
                assert len(color) == 4, (
                    f"Color array imported poorly. Each element should have length 4.")
                spots.append({
                    'pos': pos,
                    'size': size,
                    'pen': None,
                    'brush': pg.mkBrush(color[0], color[1], color[2], color[3])})
            else:
                spots.append({
                    'pos': pos,
                    'size': size,
                    'pen': None,
                    'brush': pg.mkBrush(color)})
        self.scatter_GUI.addPoints(spots)
        self.axis.addItem(self.scatter_GUI)

    def show(self):
        self.window.show()
        self.app.exec_()


def filter_and_preview_images(fns):
    # use stackFilter GUI to filter the stack of images based on contrast values
    SF = stackFilter(fns)
    SF.contrast_filter()
    eye = SF.arr
    # 1. convert to spherical coordinates by fitting a sphere with OLS
    # center the points around the mean
    eye = eye - eye.mean(0)
    # eye = np.round(eye)
    # use Points class to fit a sphere and convert to spherical coordinates, and
    eye = Points(eye, center_points=True, rotate_com=True)
    return eye


def main():
    # make a QApplication for the pyqt GUI
    app = QApplication([])
    # TODO: main menu with buttons for different steps.
    # what follows is a typical pipeline
    # if user wants to skip ahead, they have to load a particular kind of file

    # make a UI that checks for saved files for coordinates, cross_section, etc.
    filenames = [
        "eye_coordinates_Points.pkl",
        "eye_cross_section.pkl",
        "eye_cross_section_image.pkl",
        "eye_crystalline_cone_clusters.pkl",
        "eye_cone_cluster_data.csv",
        "eye_cone_pair_data.csv"
    ]

    # 0. import files and use GUI to get the images to filter the stack of images
    file_UI = fileSelector()
    file_UI.close()
    fns = file_UI.files
    folder = os.path.dirname(fns[0])
    filenames = [os.path.join(folder, fn) for fn in filenames]
    other_files = os.listdir(folder)
    other_files = [fn for fn in other_files if fn not in fns]
    other_files = [os.path.join(folder, fn) for fn in other_files]
    # use stackFilter GUI to filter the stack of images based on contrast values
    save = False
    while save is False:
        eye = None
        if filenames[0] in other_files:
            load_file = input(f"{filenames[0]} was found in the current folder. "
                              "Would you like to load this coordinates file? Press <1>"
                              " for yes or <0> for no.")
            if load_file in ["1", "y", "yes"]:
                eye = load_Points(filenames[0])
            else:
                os.remove(filenames[0])
                other_files.remove(filenames[0])
        if eye is None:
            eye = filter_and_preview_images(fns)
        # 3d scatter plot of the included coordinates
        # scatter = ScatterPlot3d(eye.pts)
        # scatter.show()
        response = input("Save and continue? Press <1> for yes or <0> to load "
                         "and filter the images again?")
        if response in ["1", "y", "yes"]:
            save = True
    eye.save(filenames[0])

    # 2. get approximate cross section of the points in spherical coordinates
    save = False
    thickness = .5
    while save is False:
        cross_section = None
        if filenames[1] in other_files:
            load_file = input(f"{filenames[1]} was found in the current folder. "
                              "Would you like to load this cross section file? Press <1>"
                              " for yes or <0> for no.")
            if load_file in ["1", "y", "yes"]:
                cross_section = load_Points(filenames[1])
            else:
                os.remove(filenames[1])
                other_files.remove(filenames[1])
        if cross_section is None:
            eye.get_polar_cross_section(thickness=thickness)
            cross_section = eye.cross_section
        scatter = ScatterPlot2d(cross_section.polar[:, :2],
                                colorvals=cross_section.radii)
        scatter.show()
        response = input("Save and continue? Press <1> for yes or <0> to extract "
                         "the cross section using a different thickness?")
        if response in ["1", "y", "yes"]:
            save = True
        else:
            thickness = input("What proportion of thickness (between 0 and 1) should "
                              "we use?")
            success = False
            while success is False:
                try:
                    thickness = float(thickness)
                    success = thickness <= 1 and thickness > 0
                except:
                    thickness = input(
                        "the response must be a number between 0 and 1")
    eye.save(filenames[0])
    cross_section.save(filenames[1])

    save = False
    pixel_length = .001
    while save is False:
        cross_section_eye = None
        if filenames[2] in other_files:
            load_file = input(f"{filenames[2]} was found in the current folder."
                              "Would you like to load this file? Press <1>"
                              " for yes or <0> for no.")
            if load_file in ["1", "y", "yes"]:
                # cross_section_eye = load_image(filenames[2])
                cross_section_eye = load_Points(filenames[2])
                img = cross_section_eye.image
            else:
                os.remove(filenames[2])
                other_files.remove(filenames[2])
        if cross_section_eye is None:
            img, (xvals, yvals) = cross_section.rasterize(
                pixel_length=pixel_length)
            # 3. use low pass filter method from Eye object in fly_eye to find centers
            # of cone clusters/ommatidia.
            # a. rasterize the image so that we can use our image processing algorithm in fly_eye
            cross_section_eye = fe.Eye(img, pixel_size=pixel_length)
            cross_section_eye.xvals, cross_section_eye.yvals = xvals, yvals
            # dilated = ndimage.morphology.binary_dilation(
            #     img, iterations=20).astype('uint8')
            # # b. make an approximate mask using the lower and upper bounds per column
            # mask = np.zeros(img.shape, dtype=bool)
            mask = img != 0
            mask = ndimage.binary_dilation(mask, iterations=2)
            # for col, col_vals in enumerate(dilated):
            #     inds = np.where(col_vals)[0]
            #     if inds.sum() > 0:
            #         first, last = inds.min(), inds.max()
            #         mask[col, first: last + 1] = True
        mask = img != 0
        mask = ndimage.binary_dilation(mask, iterations=2)
        pg.image(img)
        app.exec_()
        response = input("Save and continue? Press <1> for yes or <0> to rasterize "
                         "the image using a different pixel length?")
        if response in ["1", "y", "yes"]:
            save = True
        else:
            pixel_length = input("What pixel length should we use?")
            success = False
            while success is False:
                try:
                    pixel_length = float(thickness)
                except:
                    pixel_length = input("the response must be a number")

    with open(filenames[2], 'wb') as image_pkl:
        pickle.dump(cross_section_eye, image_pkl)

    save = False
    while save is False:
        min_facets = input("What is the fewest possible number of ommatidia?")
        while isinstance(min_facets, int) is False:
            try:
                min_facets = int(min_facets)
            except:
                min_facets = input(
                    "The number of ommatidia should be a whole number.")
        max_facets = input("What is the most possible number of ommatidia?")
        while isinstance(max_facets, int) is False:
            try:
                max_facets = int(max_facets)
            except:
                max_facets = input(
                    "The number of ommatidia should be a whole number.")
        process_centers = True
        if cross_section_eye.ommatidia is not None:
            response = input("The ommatidia centers have been processed already. "
                             "Do you want to skip this step? Press <1> to skip or "
                             "<0> to reprocess the data.")
            if response in ["1", "y", "yes", "skip"]:
                process_centers = False
        if process_centers:
            # c. use low pass algorithm from fly_eye on the rasterized image
            cross_section_eye.get_ommatidia(
                mask=mask, min_facets=min_facets, max_facets=max_facets)
        ys, xs = cross_section_eye.ommatidia
        # recenter the points
        centers = np.array(
            [xs + cross_section_eye.xvals.min(),
             ys + cross_section_eye.yvals.min()]).T

        # 4. find points around these crystalline cone approximate centers
        # a. use distance tree to find cross section coordinates closest to the centers
        # pdb.set_trace()
        tree = spatial.KDTree(centers)
        dists, inds = tree.query(cross_section.polar[:, :2])
        # b. include only those centers that have at least one nearest neighbor
        included = sorted(set(inds))
        centers = centers[included]
        # plot the points
        scatter_pts = ScatterPlot2d(
            cross_section.polar[:, :2],
            size=1,
            color=(1, 1, 1, 1))
        scatter_centers = ScatterPlot2d(
            centers,
            size=10,
            color=(red[0], red[1], red[2], 1),
            axis=scatter_pts.axis,
            window=scatter_pts.window)
        scatter_pts.show()
        response = input("Save ommatidia locations and continue?"
                         " Press <1> for yes to save or <0> to reprocess"
                         " the centers?")
        if response in ["1", "y", "yes"]:
            save = True

    with open(filenames[2], 'wb') as image_pkl:
        pickle.dump(cross_section_eye, image_pkl)

    # c. convert to the original euclidean space
    cluster_centers = []
    for ind in sorted(set(inds)):
        cluster_ = cross_section.original_pts[inds == ind]
        cluster_centers += [cluster_.mean(0)]
    cluster_centers = np.array(cluster_centers).astype('float16')
    # d. now, find clusters within the total dataset nearest the converted centers
    # using the K-means algorithm
    save = False
    clusters = None
    while save is False:
        if filenames[3] in other_files:
            load_file = input(f"{filenames[3]} was found in the current folder."
                              "Would you like to load this file? Press <1>"
                              " for yes or <0> for no.")
            if load_file in ["1", "y", "yes"]:
                with open(filenames[3], 'rb') as cluster_file:
                    clusters = pickle.load(cluster_file)
            else:
                os.remove(filenames[3])
                other_files.remove(filenames[3])
        if clusters is None:
            pts = np.round(eye.pts).astype(np.int16)
            cluster_centers = np.round(cluster_centers).astype(np.int16)
            # clusterer = cluster.KMeans(n_clusters=len(
            #     cluster_centers), init=cluster_centers).fit(pts)
            # groups = clusterer.labels_
            # clusters = []
            # for group in sorted(set(groups)):
            #     ind = group == groups
            #     cone = Points(eye.pts[ind], polar=eye.polar[ind],
            #                   center_points=False, rotate_com=False)
            #     clusters += [cone]
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=100,
                algorithm='boruvka_kdtree')
            safe_radius = np.percentile(abs(eye.residuals), 99)
            neighbors_tree = spatial.KDTree(eye.pts)
            clusters = []
            for num, center in enumerate(cluster_centers):
                i = neighbors_tree.query_ball_point(center, r=safe_radius)
                near_pts = eye.pts[i]
                near_polar = eye.polar[i]
                near_pts = np.round(near_pts).astype(int)
                near_polar = np.round(near_polar).astype(int)
                if len(near_pts) >= 100:
                    labels = clusterer.fit_predict(near_pts)
                    lbl_centers = []
                    lbl_names = sorted(set(labels))
                    for lbl in lbl_names:
                        pts = near_pts[labels == lbl]
                        lbl_centers += [pts.mean(0)]
                    lbl_centers = np.array(lbl_centers)
                    dist_tree = spatial.KDTree(lbl_centers)
                    dist, ind = dist_tree.query(center, k=1)
                    if dist <= 2:
                        lbl = labels == lbl_names[ind]
                        cone = Points(near_pts[lbl], polar=near_polar[lbl],
                                      center_points=False, rotate_com=False)
                        clusters += [cone]
                print_progress(num, len(cluster_centers))

        response = input(
            "Save and continue? Press <1> for yes or <0> to quit.")
        if response in ["1", "y", "yes"]:
            save = True
        # elif response in ["0", "quit", "q"]:
        #     return
    # e. and save as a pickled list
    with open(filenames[3], "wb") as fn:
        pickle.dump(clusters, fn)
    # f. save the clusters to a spreadsheet
    data_to_save = dict()
    cols = ['x_center', 'y_center', 'z_center', 'theta_center',
            'phi_center', 'r_center', 'children_pts', 'children_polar', 'n']
    for col in cols:
        data_to_save[col] = []
    for num, cone in enumerate(clusters):
        x_center, y_center, z_center = cone.pts.astype(float).mean(0)
        theta_center, phi_center, r_center = cone.polar.astype(
            float).mean(0)
        children_pts, children_polar = cone.pts.astype(
            float), cone.polar.astype(float)
        n = len(children_pts)
        for lbl, vals in zip(
                cols,
                [x_center, y_center, z_center, theta_center, phi_center, r_center,
                 children_pts, children_polar, n]):
            data_to_save[lbl] += [vals]
        print_progress(num, len(clusters))
    cone_cluster_data = pd.DataFrame.from_dict(data_to_save)
    cone_cluster_data.to_csv(filenames[4])

    # 5. Using our set of cone clusters, and the curvature implied by nearest cones,
    # we can take measurements relevant to the eye's optics.
    save = False
    while save is False:
        cones = clusters
        cone_centers = np.array([cone.pts.mean(0) for cone in cones])
        processed = [hasattr(cone, "skewness") for cone in cones]
        process_cones = True
        if all(processed):
            response = input("The clusters have been processed already. "
                             "Do you want to skip this step? Press <1> to skip or "
                             "<0> to reprocess the data.")
            if response in ["1", "y", "yes", "skip"]:
                process_cones = False
        if process_cones:
            nearest_neighbors = spatial.KDTree(cone_centers)
            dists, lbls = nearest_neighbors.query(cone_centers, k=13)
            # grab points adjacent to the center point by:
            # 1. grab 12 points nearest the center point
            # 2. cluster the points into 2 groups, near and far
            # 3. filter out the far group

            clusterer = cluster.KMeans(2, init=np.array(
                [0, 100]).reshape(-1, 1), n_init=1)
            for lbl, (center, cone) in enumerate(zip(cone_centers, cones)):
                neighborhood = cone_centers[lbls[lbl]]
                neighbor_dists = dists[lbl].reshape(-1, 1)
                neighbor_groups = clusterer.fit_predict(neighbor_dists[1:])
                cone.neighbor_lbls = lbls[lbl][1:][neighbor_groups == 0]

                # approximate lens diameter using distance to nearest neighbors
                diam = np.mean(neighbor_dists[1:][neighbor_groups == 0])
                area = np.pi * (.5 * diam) ** 2
                cone.lens_area = area

                # approximate ommatidial axis vector by referring to the center of a
                # sphere fitting around the nearest neighbors
                pts = Points(neighborhood, center_points=False)
                pts.spherical()
                d_vector = pts.center - center
                d_vector /= LA.norm(d_vector)
                cone.approx_vector = d_vector

                # approximate ommatidial axis vector by regressing cone data
                d_vector2 = cone.get_line()
                cone.anatomical_vector = d_vector2

                # calculate the anatomical skewness (angle between the two vectors)
                inside_ang = min(
                    angle_between(d_vector, d_vector2),
                    angle_between(d_vector, -d_vector2))
                cone.skewness = inside_ang

                print_progress(lbl, len(cone_centers))

        lens_area = np.array([cone.lens_area for cone in cones])
        anatomical_vectors = np.array(
            [cone.anatomical_vector for cone in cones])
        approx_vectors = np.array([cone.approx_vector for cone in cones])
        skewness = np.array([cone.skewness for cone in cones])
        neighbor_lbls = np.array([cone.neighbor_lbls for cone in cones])

        # TODO: 3 scatter plots showing the three parameters
        scatter_lens_area = ScatterPlot3d(
            cone_centers,
            size=10,
            colorvals=lens_area)
        scatter_lens_area.show()
        scatter_skewness = ScatterPlot3d(
            cone_centers,
            size=10,
            colorvals=skewness)
        scatter_skewness.show()
        response = input(
            "Save and continue? Press <1> for yes, <0> to reprocess, or <q> to quit.")
        if response in ["1", "y", "yes"]:
            save = True
        # elif response in ["quit", "q"]:
        #     return

    cone_cluster_data['lens_area'] = lens_area
    cone_cluster_data['anatomical_axis'] = anatomical_vectors.tolist()
    cone_cluster_data['approx_axis'] = approx_vectors.tolist()
    cone_cluster_data['skewness'] = skewness
    cone_cluster_data.to_csv(filenames[4])
    with open(filenames[3], 'wb') as pickle_file:
        pickle.dump(clusters, pickle_file)

    # 6. using the cone vectors, we can also calculate inter-ommatidial
    # angles by finding the minimum angle difference between adjacent
    # cone pairs
    pairs_tested = set()
    interommatidial_angle_approx = dict()
    interommatidial_angle_anatomical = dict()
    for num, cone in enumerate(cones):
        approx_IOAs = []
        anatomical_IOAs = []
        pairs = []
        for neighbor in cone.neighbor_lbls:
            pair = tuple(sorted([num, neighbor]))
            if pair not in pairs_tested:
                neighbor_cone = cones[neighbor]
                position1, direction1 = (
                    cone.center, cone.anatomical_vector)
                position2, direction2 = (
                    neighbor_cone.center, neighbor_cone.anatomical_vector)
                anatomical_angle = angle_between_skew_vectors(
                    position1, direction1,
                    position2, direction2)
                direction1, direction2 = (
                    cone.approx_vector, neighbor_cone.approx_vector)
                approx_angle = angle_between_skew_vectors(
                    position1, direction1,
                    position2, direction2)
                interommatidial_angle_anatomical[pair] = anatomical_angle
                interommatidial_angle_approx[pair] = approx_angle
                pairs_tested.add(pair)
                pairs += [pair]
            else:
                anatomical_angle = interommatidial_angle_anatomical[pair]
                approx_angle = interommatidial_angle_approx[pair]
            anatomical_IOAs += [anatomical_angle]
            approx_IOAs += [approx_angle]
        cone.approx_FOV = np.mean(approx_IOAs)
        cone.anatomical_FOV = np.mean(anatomical_IOAs)
        print_progress(num, len(cones))

    with open(filenames[3], 'wb') as pickle_file:
        pickle.dump(clusters, pickle_file)

    import pdb
    pdb.set_trace()

    pairs_tested = np.array(list(pairs_tested))
    IOA_approx = np.array(list(interommatidial_angle_approx.values()))
    IOA_anatomical = np.array(list(interommatidial_angle_anatomical.values()))
    data_to_save = dict()
    cols = ['cluster1', 'cluster2', 'cluster1_center', 'cluster2_center',
            'cluster1_polar_center', 'cluster2_polar_center',
            'approx_angle', 'anatomical_angle']
    for col in cols:
        data_to_save[col] = []
    for num, (pair, approx, anatomical) in enumerate(
            zip(pairs_tested, IOA_approx, IOA_anatomical)):
        ind1, ind2 = pair
        cluster1, cluster2 = clusters[ind1], clusters[ind2]
        for lbl, vals in zip(
                cols,
                [ind1, ind2, cluster1.pts.mean(0), cluster2.pts.mean(0),
                 cluster1.polar.mean(0), cluster2.polar.mean(0),
                 approx, anatomical]):
            data_to_save[lbl] += [vals]
        print_progress(num, len(pairs_tested))
    cone_pair_data = pd.DataFrame.from_dict(data_to_save)
    cone_pair_data.to_csv(filenames[5])


if __name__ == "__main__":
    main()
