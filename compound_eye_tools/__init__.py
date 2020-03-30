from matplotlib import pyplot as plt
from scipy import spatial, interpolate, ndimage
import numpy as np
import PIL
import sys
import os
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


def save_image(fn, arr):
    """Save an image using the PIL."""
    img = PIL.Image.fromarray(arr)
    if os.path.exists(fn):
        os.remove(fn)
    return img.save(fn)


def print_progress(part, whole, bar=True):
    """Print the part/whole progress bar."""
    prop = float(part)/float(whole)
    sys.stdout.write('\r')
    # sys.stdout.write("[%-20s] %d%%" % ("="*int(20*prop), round(100)*prop))
    if bar:
        prog_bar = '='*int(20*prop)
        st = f"[{prog_bar:20s}] {round(100 * prop)}%"
    else:
        st = f"{round(100*prop)}%\n"
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

    def rasterize(self, mode='polar', axes=[0, 1], pixel_length=.01,
                  image_size=10**5):
        """Rasterize float coordinates in to a grid defined by min and max vals
        and sampled at pixel_length.
        """
        if mode == 'pts':
            arr = self.pts
        if mode == 'polar':
            arr = self.polar
        x, y, z = arr.T

        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        # figure out side lengths needed for input image size
        ratio = y_range / x_range
        x_len = int(np.round(np.sqrt(image_size/ratio)))
        y_len = int(np.round(ratio * x_len))
        # get x and y ranges corresponding to image size
        xs = np.linspace(x.min(), x.max(), x_len)
        ys = np.linspace(y.min(), y.max(), y_len)
        avg = []
        for col_num, (x1, x2) in enumerate(zip(xs[:-1], xs[1:])):
            in_column = np.logical_and(x >= x1, x < x2)
            in_column = arr[in_column]
            for row_num, (y1, y2) in enumerate(zip(ys[:-1], ys[1:])):
                in_row = np.logical_and(
                    in_column[:, 1] >= y1, in_column[:, 1] < y2)
                avg += [in_row.sum()]
            print_progress(col_num, len(xs) - 1)
        print("\n")
        avg = np.array(avg)
        avg = avg.reshape((len(xs) - 1, len(ys) - 1))
        self.raster = avg
        xs = xs[:-1] + (pixel_length / 2.)
        ys = ys[:-1] + (pixel_length / 2.)
        return self.raster

    # def fit_surface(self, mode='polar', outcome_axis=0, pixel_length=.01):
    def fit_surface(self, mode='polar', outcome_axis=0, image_size=10**3):
        """Find cubic interpolation surface of one axis using the other two."""
        if mode == 'pts':
            arr = self.pts
        if mode == 'polar':
            arr = self.polar
        x, y, z = arr.T
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        # figure out side lengths needed for input image size
        ratio = y_range / x_range
        x_len = int(np.round(np.sqrt(image_size/ratio)))
        y_len = int(np.round(ratio * x_len))
        # reduce data using a 2D rolling average
        # xs = np.arange(x.min(), x.max(), pixel_length)
        # ys = np.arange(y.min(), y.max(), pixel_length)
        xs = np.linspace(x.min(), x.max(), x_len)
        ys = np.linspace(y.min(), y.max(), y_len)
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
        print("\n")
        avg = np.array(avg)
        # filter outlier points by using bootstraped 95% confidence band (not of the mean)
        low, high = np.percentile(avg[:, 2], [.5, 99.5])
        avg = avg[np.logical_and(avg[:, 2] >= low, avg[:, 2] < high)]
        self.avg_raster = avg
        avg_x, avg_y, avg_z = avg.T
        # interpolate through avg to get 'cross section' using
        # bivariate spline (bisplrep)
        # tck = interpolate.bisplrep(avg_x, avg_y, avg_z, s=0)
        # interp_func = interpolate.interp2d(avg_x, avg_y, avg_z, kind='cubic')
        # interp_func = interpolate.LinearNDInterpolator(avg[:, :2], avg[:, 2])
        interp_func = interpolate.LinearNDInterpolator(avg[:, :2], avg[:, 2])
        # import pdb
        # pdb.set_trace()
        z_new = interp_func(x, y)
        no_nans = np.isnan(z_new) == False
        self.pts = self.pts[no_nans]
        self.polar = self.polar[no_nans]
        self.x, self.y, self.z = self.pts.T
        self.theta, self.phi, self.radii = self.polar.T
        self.surface = z_new[no_nans]

    def get_polar_cross_section(self, thickness=.1, pixel_length=.01):
        """Find best fitting surface of radii using phis and thetas."""
        # self.fit_surface(mode='polar', pixel_length=pixel_length)
        self.fit_surface(mode='polar')
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
        self.latest_fn = fn
        with open(fn, "wb") as pickle_file:
            pickle.dump(self, pickle_file)


filetypes = [
    ('tiff images', '*.tiff *.TIFF *.tff *.TFF *.tif *.TIF'),
    ('jpeg images', '*.jpeg *.jpg *.JPEG *.JPG'),
    ('png images', '*.png *.PNG'),
    ('bmp images', '*.bmp *.BMP'),
    ('all types', '*.*')]

ftypes = [f"{fname} ({ftype})" for (fname, ftype) in filetypes]
ftypes = ";;".join(ftypes)


class fileSelector(QWidget):
    """Offer a file selection dialog box filtering common image filetypes."""

    def __init__(self, filetypes=ftypes, app=None,
                 title='Select the images you want to process.'):
        if app is None:
            self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        super().__init__()
        self.title = title
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
            self.title,
            "",
            self.filetypes,
            options=options)


class folderSelector(QWidget):
    """Offer a directory selection dialog box."""

    def __init__(self, filetypes=ftypes, app=None,
                 title='Select a folder'):
        if app is None:
            self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        super().__init__()
        self.title = title
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.filetypes = filetypes
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.openDialog()
        self.show()

    def openDialog(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        self.folder = QFileDialog.getExistingDirectory(
            self,
            self.title,
            "",
            options=options)


class stackFilter():
    """Import image filenames filter images using upper and lower contrast bounds."""

    def __init__(self, fns=os.listdir("./"), app=None):
        """Import images using fns, a list of filenames."""
        if app is None:
            self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        self.fns = fns
        self.folder = os.path.dirname(self.fns[0])
        imgs = []
        print("Loading images:\n")
        for num, fn in enumerate(fns):
            try:
                imgs += [load_image(fn)]
            except:
                print(f"{fn} failed to load.")
                imgs += [np.zeros(imgs[-1].shape)]
            print_progress(num, len(fns))
        print("\n")
        assert len(imgs) > 0, "All images failed to load."
        self.imgs = np.array(imgs, dtype=np.uint16)

    def contrast_filter(self):
        """Use pyqtgraph's image UI to select lower an upper bound contrasts."""
        # if there is no defined application instance, make one
        if self.app is None:
            self.app = QApplication.instance()
            if self.app is None:
                self.app = QApplication([])
        # self.image_view = pg.ImageView()
        # self.image_view.show()
        # self.image_view.setImage(self.imgs)
        # self.window = pg.GraphicsLayoutWidget()
        # self.image_UI = self.window.addPlot()
        self.image_UI = pg.image(self.imgs[:10].astype('uint8'))  
        # use the image UI from pyqtgraph
        self.image_UI.setPredefinedGradient('greyclip')
        self.app.exec_()      # allow the application to run until closed
        # grab low and high bounds from UI
        self.low, self.high = self.image_UI.getLevels()
        # xs = np.array([], dtype='uint16')
        # ys = np.array([], dtype='uint16')
        # zs = np.array([], dtype='uint16')

        print("Extracting coordinate data: ")
        np.logical_and(self.imgs <= self.high, self.imgs > self.low,
                       out=self.imgs)
        self.imgs = self.imgs.astype(bool, copy=False)
        try:
            ys, xs, zs = np.where(self.imgs)
        except:
            print(
                "coordinate data is too large. Using a hard drive memory map instead of RAM.")
            self.imgs_memmap = np.memmap(os.path.join(self.folder, "volume.npy"),
                                         mode='w+', shape=self.imgs.shape, dtype=bool)
            self.imgs_memmap[:] = self.imgs[:]
            del self.imgs
            self.imgs = None
            for depth, img in enumerate(self.imgs_memmap):
                y, x = np.where(img)
                z = np.repeat(depth, len(x))
                xs = np.append(xs, x)
                ys = np.append(ys, y)
                zs = np.append(zs, z)
                print_progress(depth + 1, len(self.imgs))
        self.arr = np.array([xs, ys, zs], dtype=np.uint16).T

    def pre_filter(self):
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
        dirname = os.path.dirname(self.fns[0])
        folder = os.path.join(dirname, 'prefiltered')
        if not os.path.isdir(folder):
            os.mkdir(folder)
        np.logical_and(self.imgs > self.low, self.imgs <
                       self.high, out=self.imgs)
        self.imgs = self.imgs.astype('uint8', copy=False)
        np.multiply(self.imgs, 255, out=self.imgs)
        print("Saving filtered images:\n")
        for num, (fn, img) in enumerate(zip(self.fns, self.imgs)):
            base = os.path.basename(fn)
            new_fn = os.path.join(folder, base)
            save_image(new_fn, img)
            print_progress(num + 1, len(self.fns))


class ScatterPlot3d():
    """Plot 3d datapoints using pyqtgraph's GLScatterPlotItem."""

    def __init__(self, arr, color=None, size=1, app=None, window=None,
                 colorvals=None, cmap=plt.cm.viridis):
        self.arr = arr
        self.app = app
        if self.app is None:
            self.app = QApplication.instance()
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
        if colorvals is not None:
            assert len(colorvals) == self.n, print("input colorvals should "
                                                   "have the same lengths as "
                                                   "input array")
            if np.any(colorvals < 0) or np.any(colorvals > 1):
                colorvals = (colorvals - colorvals.min()) / \
                    (colorvals.max() - colorvals.min())
            self.color = np.array([self.cmap(c) for c in colorvals])
        elif color is not None:
            assert len(color) == 4, print("color input should be a list or tuple "
                                          "of RGBA values between 0 and 1")
            if isinstance(self.color, (tuple, list)):
                self.color = np.array(self.color)
            if self.color.max() > 1:
                self.color = self.color / self.color.max()
            self.color = tuple(self.color)
        else:
            self.color = (1, 1, 1, 1)
        self.plot()

    def plot(self):
        if self.window is None:
            self.window = gl.GLViewWidget()
            self.window.setWindowTitle("3D Scatter Plot")
        self.scatter_GUI = gl.GLScatterPlotItem(
            pos=self.arr, size=self.size, color=self.color)
        self.window.addItem(self.scatter_GUI)

    def show(self):
        if self.app is None:
            self.app = QApplication([])
        self.window.show()
        self.app.exec_()


class ScatterPlot2d():
    """Plot 2d datapoints using pyqtgraph's GLScatterPlotItem."""

    # def __init__(self, arr, color=(1, 1, 1, 1), size=[1], app=None, window=None):
    def __init__(self, arr, color=None, app=None, window=None, size=1,
                 axis=None, colorvals=None, cmap=plt.cm.viridis, scatter_GUI=None):

        if app is None:
            self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
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
        if self.color is None and colorvals is not None:
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


def make_directory(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


class Benchmark():
    def __init__(self, filename, function, folder=False, name=None):
        self.filename = filename
        self.function = function
        self.folder = folder
        self.name = name
        if self.name is None:
            self.name = os.path.basename(self.filename)
        if self.folder:
            self.statement = f"{self.name} Would you like to load files from {os.path.basename(self.filename)} and continue?"
        else:
            self.statement = f"{self.name} Would you like to load {os.path.basename(self.filename)} and continue?"

    def exists(self):
        if self.folder:
            if os.path.isdir(self.filename):
                fns = os.listdir(self.filename)
                return len(fns) > 1
            else:
                return False
        else:
            if os.path.exists(self.filename):
                return os.stat(self.filename).st_size > 0
            else:
                return False

    def load(self):
        if self.folder:
            return self.filename
        elif self.filename.endswith(".npy"):
            return np.load(self.filename)
        elif self.filename.endswith(".points"):
            return load_Points(self.filename)
        elif self.filename.split(".")[-1] in img_filetypes:
            return load_image(self.filename)
        elif self.filename.endswith(".csv"):
            return pd.read_csv(self.filename)


def init():
    # choose your project directory
    folder_UI = folderSelector(
        title='Choose the folder containing the original image stack: ')
    folder_UI.close()
    home_folder = folder_UI.folder
    project_folder = os.path.join(home_folder, "compound_eye_data")
    make_directory(project_folder)
    custom_filenames = [
        "prefiltered_stack",
        "coordinates.points",
        "cross_section_raster.png",
        "ommatidia_centers.points",
        "ommatidia_clusters",
        "ommatidia_measurements.csv",
        "inter_ommatidial_measurements.csv",
        "whole_eye_measurements.csv"]
    custom_names = [
        'Image stack was prefiltered.',
        '3D coordinates were processed.',
        'The cross section was rasterized.',
        'Ommatidia centers were extracted.',
        'Ommatidia clusters were found.',
        'Ommatidial measurements were calculated.',
        'Inter-ommatidial measurements were calculated.',
        'Whole eye measurements were calculated.']
    functions = [
        pre_filter,             # optional
        import_stack,           # works even if no prefilter
        get_cross_section,
        find_ommatidia_centers,
        find_ommatidia_clusters,
        get_ommatidial_measurements,
        get_interommatidial_measurements,
        get_whole_eye_measurements,
        close]
    folders = ["." not in os.path.basename(fn) for fn in custom_filenames]
    benchmarks = []
    for filename, function, folder, name in zip(custom_filenames,
                                                functions,
                                                folders,
                                                custom_names):
        if folder:
            benchmark_fn = os.path.join(home_folder, filename)
        else:
            benchmark_fn = os.path.join(project_folder, filename)
        benchmarks.append(
            Benchmark(benchmark_fn, function,
                      folder, name))
    progress = 0
    benchmarks_present = [mark for mark in benchmarks if mark.exists()]
    # check what is the last benchmark done; start from there?
    # if no, offer all accomplished benchmarks to start from
    # this is nice because it will be the start screen for new projects
    print("The following benchmarks were found:")
    for num, mark in enumerate(benchmarks_present):
        print(f"{num+1}. {mark.statement}")
    choice = None
    while choice not in np.arange(len(functions)).astype(str):
        choice = input(
            "Press the number to continue from that benchmark or press <0> to go to the main menu: ")
    choice = int(choice)
    if choice == 0:
        print("Main Menu\n"
              "0. Pre-filter an image stack\n"
              "1. Import and filter an image stack as is\n")
        choice = None
        while choice not in ['0', '1']:
            choice = input(
                "Choose by entering the number to the left: ")
        if choice == '0':
            pre_filter(home_folder)
            return
    progress = int(choice)
    functions = functions[progress:]
    benchmark = benchmarks[progress - 1].load()
    print("Would you like to preview the outcome of each step? ")
    preview = None
    while preview not in ['0', '1']:
        preview = input("Press <1> for yes and <0> for no. ")
    preview = preview == '1'
    input_val = benchmark
    for function in functions:
        input_val = function(input_val, preview)


def pre_filter(home_folder):
    fns = os.listdir(home_folder)
    SF = stackFilter(fns)
    SF.pre_filter()
    return home_folder


img_filetypes = ['tiff', 'tff', 'tif', 'jpeg', 'jpg', 'png', 'bmp']


def import_stack(folder, preview=True):
    # return coordinates as a Points object
    basename = os.path.basename(folder)
    if basename == 'prefiltered_stack':
        prefilter_folder = folder
        home_folder = os.path.dirname(folder)
    else:
        home_folder = folder
        prefilter_folder = os.path.join(folder, "prefiltered_stack")
    if os.path.isdir(prefilter_folder):
        fns = os.listdir(prefilter_folder)
        fns = [fn for fn in fns if fn.split(".")[-1].lower() in img_filetypes]
        assert len(fns) > 1, print(
            "The folder prefilter_stack has no image files.")
        filenames = [os.path.join(prefilter_folder, fn) for fn in fns]
    else:
        fns = os.listdir(home_folder)
        fns = [fn for fn in fns if fn.split(".")[-1].lower() in img_filetypes]
        assert len(fns) > 1, print(
            f"The home folder, {home_folder}, has no image files.")
        filenames = [os.path.join(home_folder, fn) for fn in fns]
    filenames.sort()
    # use stackFilter GUI to filter the stack of images based on contrast values
    save = False
    while save is False:
        eye = None
        if eye is None:
            eye = filter_and_preview_images(filenames)
        # 3d scatter plot of the included coordinates
        if preview:
            scatter = ScatterPlot3d(eye.pts)
            scatter.show()
        print("Save and continue? ")
        response = None
        while response not in ['0', '1']:
            response = input("Press <1> for yes or <0> to load "
                             "and filter the images again? ")
        save = response == '1'
    eye.save(os.path.join(home_folder, "compound_eye_data", "coordinates.points"))
    return eye


def get_cross_section(eye, preview=True, thickness=.8):
    save = False
    while save is False:
        eye.get_polar_cross_section(thickness=thickness)
        cross_section = eye.cross_section
        # this part is very slow. can we perform it on just a subset of the points?
        if preview:
            img = cross_section.rasterize()
            plt.imshow(img)
            plt.show()
        response = None
        print("Save and continue? ")
        while response not in ['0', '1']:
            response = input("Press <1> for yes or <0> to extract "
                             "the cross section using a different thickness?")
        if response == "1":
            save = True
        else:
            thickness = inf
            while not np.logical_and(thickness >= 0, thickness <= 1):
                thickness = input("What proportion of thickness (from 0 to 1) should "
                                  "we use?")
                try:
                    thickness = float(thickness)
                except:
                    thickness = inf
    eye.save(eye.latest_fn)
    project_folder = os.path.basename(eye.latest_fn)
    save_image(cross_section,
               os.path.join(project_folder, "cross_section_raster.png"))
    return cross_section


def find_ommatidia_centers(cross_section_points, preview=True):
    return


def find_ommatidia_clusters():
    return


def get_ommatidial_measurements():
    return


def get_interommatidial_measurements():
    return


def get_whole_eye_measurements():
    return


def close():
    return


def main():
    # make a QApplication for the pyqt GUI
    # TODO: main menu with buttons for different steps.
    # what follows is a typical pipeline
    # if user wants to skip ahead, they have to load a particular kind of file

    # make a UI that checks for saved files for coordinates, cross_section, etc.
    filenames = [
        "eye_coordinates_Points.pkl",
        "eye_cross_section.pkl",
        "eye_cross_section_image.pkl",
        "cluster_centers.npy",
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
    thickness = .8
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
        # scatter = ScatterPlot2d(cross_section.polar[:, :2],
        #                         colorvals=cross_section.radii)
        # scatter.show()
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
            mask = img != 0
            mask = ndimage.binary_dilation(mask, iterations=20)
            mask = ndimage.binary_erosion(mask, iterations=20)
            cross_section_eye = fe.Eye(img, pixel_size=pixel_length, mask=mask)
            cross_section_eye.xvals, cross_section_eye.yvals = xvals, yvals
            # dilated = ndimage.morphology.binary_dilation(
            #     img, iterations=20).astype('uint8')
            # # b. make an approximate mask using the lower and upper bounds per column
            # mask = np.zeros(img.shape, dtype=bool)
            # for col, col_vals in enumerate(dilated):
            #     inds = np.where(col_vals)[0]
            #     if inds.sum() > 0:
            #         first, last = inds.min(), inds.max()
            #         mask[col, first: last + 1] = True
        # mask = img != 0
        # mask = ndimage.binary_dilation(mask, iterations=2)
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
                    pixel_length = float(pixel_length)
                    success = True
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
            # cross_section_eye.mask = mask
            cross_section_eye.get_ommatidia(
                min_facets=min_facets, max_facets=max_facets, method=0)
            breakpoint()
        ys, xs = cross_section_eye.ommatidia
        ps = cross_section_eye.pixel_size
        plt.imshow(img, cmap='gray')
        plt.scatter(ys/ps, xs/ps, marker='.', color=red)
        plt.show()
        # recenter the points
        centers = np.array(
            [xs + cross_section_eye.xvals.min(),
             ys + cross_section_eye.yvals.min()]).T

        # 4. find points around these crystalline cone approximate centers
        # a. use distance tree to find cross section coordinates closest to the centers
        # pdb.set_trace()
        # tree = spatial.KDTree(centers)
        # dists, inds = tree.query(cross_section.polar[:, :2])
        # b. include only those centers that have at least one nearest neighbor
        # included = sorted(set(inds))
        # centers = centers[included]
        # plot the points
        # scatter_pts = ScatterPlot2d(
        #     cross_section.polar[:, :2],
        #     size=1,
        #     color=(1, 1, 1, 1))
        # scatter_centers = ScatterPlot2d(
        #     centers,
        #     size=10,
        #     color=(red[0], red[1], red[2], 1),
        #     axis=scatter_pts.axis,
        #     window=scatter_pts.window)
        # scatter_pts.show()
        response = input("Save ommatidia locations and continue?"
                         " Press <1> for yes to save or <0> to reprocess"
                         " the centers?")
        if response in ["1", "y", "yes"]:
            save = True

    with open(filenames[2], 'wb') as image_pkl:
        pickle.dump(cross_section_eye, image_pkl)

    # c. convert to the original euclidean space
    cluster_centers = None
    if filenames[3] in other_files:
        load_file = input(f"{filenames[3]} was found in the current folder."
                          "Would you like to load this file? Press <1>"
                          " for yes or <0> for no.")
        if load_file in ["1", "y", "yes"]:
            cluster_centers = np.load(filenames[3])
        else:
            os.remove(filenames[3])
            other_files.remove(filenames[3])
    if cluster_centers is None:
        tree = spatial.KDTree(centers)
        # for center in centers:
        inds = []
        total = len(cross_section.polar[:, :2])
        for num, p in enumerate(cross_section.polar[:, :2]):
            dist, ind = tree.query(p, k=1)
            inds.append(ind)
            print_progress(num, total)
        print("\n")
        inds = np.array(inds)
        cluster_centers = []
        clusters = []
        for ind in sorted(set(inds)):
            cluster_ = cross_section.original_pts[inds == ind]
            clusters.append(cluster_)
            cluster_centers += [cluster_.mean(0)]
        cluster_centers = np.array(cluster_centers).astype('float16')
        np.save(filenames[3], cluster_centers)
    # d. now, find clusters within the total dataset nearest the converted centers
    # using a cluster algorithm
    save = False
    clusters = None
    while save is False:
        if filenames[4] in other_files:
            load_file = input(f"{filenames[4]} was found in the current folder."
                              "Would you like to load this file? Press <1>"
                              " for yes or <0> for no.")
            if load_file in ["1", "y", "yes"]:
                with open(filenames[4], 'rb') as cluster_file:
                    clusters = pickle.load(cluster_file)
            else:
                os.remove(filenames[4])
                other_files.remove(filenames[4])
        if clusters is None:
            pts = np.round(eye.pts).astype(np.int16)
            cluster_centers = np.round(cluster_centers).astype(np.int16)
            clusterer = cluster.KMeans(n_clusters=len(
                cluster_centers), init=cluster_centers).fit(pts)
            groups = clusterer.labels_
            clusters = []
            for group in sorted(set(groups)):
                ind = group == groups
                cone = Points(eye.pts[ind], polar=eye.polar[ind],
                              center_points=False, rotate_com=False)
                clusters += [cone]
            # clusterer = hdbscan.HDBSCAN(
            #     min_cluster_size=100,
            #     algorithm='boruvka_kdtree')
            # safe_radius = np.percentile(abs(eye.residuals), 99)
            # neighbors_tree = spatial.KDTree(eye.pts)
            # clusters = []
            # for num, center in enumerate(cluster_centers):
            #     i = neighbors_tree.query_ball_point(center, r=safe_radius)
            #     near_pts = eye.pts[i]
            #     near_polar = eye.polar[i]
            #     near_pts = np.round(near_pts).astype(int)
            #     near_polar = np.round(near_polar).astype(int)
            #     if len(near_pts) >= 100:
            #         labels = clusterer.fit_predict(near_pts)
            #         lbl_centers = []
            #         lbl_names = sorted(set(labels))
            #         for lbl in lbl_names:
            #             pts = near_pts[labels == lbl]
            #             lbl_centers += [pts.mean(0)]
            #         lbl_centers = np.array(lbl_centers)
            #         dist_tree = spatial.KDTree(lbl_centers)
            #         dist, ind = dist_tree.query(center, k=1)
            #         if dist <= 2:
            #             lbl = labels == lbl_names[ind]
            #             cone = Points(near_pts[lbl], polar=near_polar[lbl],
            #                           center_points=False, rotate_com=False)
            #             clusters += [cone]
            #     print_progress(num, len(cluster_centers))

        response = input(
            "Save and continue? Press <1> for yes or <0> to quit.")
        if response in ["1", "y", "yes"]:
            save = True
        # elif response in ["0", "quit", "q"]:
        #     return
    # e. and save as a pickled list
    with open(filenames[4], "wb") as fn:
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
    print("\n")
    cone_cluster_data = pd.DataFrame.from_dict(data_to_save)
    cone_cluster_data.to_csv(filenames[5])

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
    cone_cluster_data.to_csv(filenames[5])
    with open(filenames[4], 'wb') as pickle_file:
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
    print("\n")
    with open(filenames[4], 'wb') as pickle_file:
        pickle.dump(clusters, pickle_file)

    # import pdb
    # pdb.set_trace()

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
    print("\n")
    cone_pair_data = pd.DataFrame.from_dict(data_to_save)
    cone_pair_data.to_csv(filenames[4])


if __name__ == "__main__":
    init()

# eye = load_Points(
#     "../../13460_tiff-stack-volume/prefiltered/eye_coordinates_Points.pkl")
# eye.get_polar_cross_section(thickness=.1)

# main()
