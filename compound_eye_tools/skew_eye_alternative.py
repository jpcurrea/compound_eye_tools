import math
import numpy as np
from numpy import linalg as LA
import os
import pandas as pd
import pickle
import PIL
from scipy import spatial, interpolate, ndimage, optimize
from skimage.morphology import convex_hull_image
import sys

# graphic libraries
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.backend_bases import NavigationToolbar2
from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from sklearn import cluster

import fly_eye as fe

blue, green, yellow, orange, red, purple = [(0.30, 0.45, 0.69), (0.33, 0.66, 0.41), (
    0.83, 0.74, 0.37), (0.78, 0.50, 0.16), (0.77, 0.31, 0.32), (0.44, 0.22, 0.78)]

if 'app' not in globals():
    app = QApplication([])

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


def fit_line(data):             # fit 3d line to 3d data
    """Use singular value decomposition (SVD) to find the best fitting vector to the data.


    Keyword arguments:
    data -- input data points
    component -- the order of the axis used to decompose the data (default 0 => the first component vector which represents the plurality of the data
    """

    m = data.mean(0)
    max_val = np.round(2*abs(data - m).max()).astype(int)
    uu, dd, vv = np.linalg.svd(data - m)
    return vv


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


def spherical_to_cartesian(polar, center=np.array([0, 0, 0])):
    """Convert rectangular/cartesian to spherical coordinates."""
    theta, phi, radii = polar.T
    phi -= np.pi
    xs = radii * np.sin(theta) * np.cos(phi)
    ys = radii * np.sin(theta) * np.sin(phi)
    zs = radii * np.cos(theta)
    pts = np.array([xs, ys, zs]).T
    pts += center
    return pts


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def intersection(pos1, dir1, pos2, dir2):
    """ Returns intersection point given two 3D vectors, 
    assuming it must intersect."""
    # test values
    # pos1 = np.array([6, 8, 4])
    # pos2 = np.array([6, 8, 2])
    # dir1 = -np.array([6, 7, 0])
    # dir2 = -np.array([6, 7, 4])
    # find vector between the two position vectors
    dist = pos2 - pos1
    # the offset is given by a proportion of cross products
    offset = np.linalg.norm(np.cross(dir2, dist))/(np.linalg.norm(np.cross(dir1, dir2))) * dir1
    # the direction of the offset is positive if the two cross products are in the same direction 
    positive = np.all((np.cross(dir2, dist) > 0) == (np.cross(dir2, dir1) > 0))
    if not positive:
        offset *= -1
    return pos1 + offset


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
    theta = angle_between(centered_p1, centered_p2)
    return theta


def skew_vector_intersection(
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
    # centered_p1, centered_p2 = p1 - midpoint, p2 - midpoint
    # c1, c2 = centered_p1, centered_p2
    # theta = np.arccos(
    #     np.dot(c1, c2)/(LA.norm(c1) * LA.norm(c2)))
    return midpoint


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
                 rotate_com=True, vals=None, interp_func=None):
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
        self.vals = vals
        self.shape = self.pts.shape
        self.center = np.array([0, 0, 0])
        self.polar = None
        self.raster = None
        self.xvals, self.yvals = None, None
        self.interp_func = interp_func
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
        if self.vals is None:
            out = Points(self.pts[key], polar=self.polar[key],
                         rotate_com=False, spherical_conversion=False)
        else:
            out = Points(self.pts[key], polar=self.polar[key],
                         rotate_com=False, spherical_conversion=False,
                         vals=self.vals[key])
        out.residuals = self.residuals[key]
        out.original_pts = self.original_pts
        out.interp_func = self.interp_func
        out.surface = self.surface[key]
        return out

    def spherical(self, center=None):
        if center is None:
            center = self.center
        self.polar = cartesian_to_spherical(self.pts, center=center)
        self.theta, self.phi, self.radii = self.polar.T
        self.residuals = self.radii - self.radius

    def get_line(self):
        return fit_line(self.pts)

    def rasterize(self, mode='polar', axes=[0, 1], image_size=10**5):
        """Rasterize float coordinates in to a grid defined by min and max vals
        and sampled at pixel_length.
        """
        if mode == 'pts':
            arr = self.pts
        if mode == 'polar':
            arr = self.polar
        x, y, z = arr.T
        vals = self.vals
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        # figure out side lengths needed for input image size
        ratio = y_range / x_range
        # x_len = int(np.round(np.sqrt(image_size/ratio)))
        # y_len = int(np.round(ratio * x_len))
        x_len = int(np.round(np.sqrt(image_size/ratio)))
        # get x and y ranges corresponding to image size
        xs = np.linspace(x.min(), x.max(), x_len)
        self.raster_pixel_length = xs[1] - xs[0]
        ys = np.arange(y.min(), y.max(), self.raster_pixel_length)
        # avg = []
        # for col_num, (x1, x2) in enumerate(zip(xs[:-1], xs[1:])):
        #     in_column = np.logical_and(x >= x1, x < x2)
        #     in_column = arr[in_column]
        #     for row_num, (y1, y2) in enumerate(zip(ys[:-1], ys[1:])):
        #         in_row = np.logical_and(
        #             in_column[:, 1] >= y1, in_column[:, 1] < y2)
        #         avg += [in_row.sum()]
        #     print_progress(col_num, len(xs) - 1)
        # print("\n")
        # avg = np.array(avg)
        # avg = avg.reshape((len(xs) - 1, len(ys) - 1))
        if vals is None:
            avg = np.histogram2d(x, y, bins=(xs, ys))
        else:
            avg = np.histogram2d(x, y, bins=(xs, ys), weights = vals)
        avg = avg[0]
        self.raster = avg
        xs = xs[:-1] + (self.raster_pixel_length / 2.)
        ys = ys[:-1] + (self.raster_pixel_length / 2.)
        self.xvals, self.yvals = xs, ys
        return self.raster, (xs, ys)

    # def fit_surface(self, mode='polar', outcome_axis=0, pixel_length=.01):
    def fit_surface(self, mode='polar', image_size=10**3):
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
        self.interp_func = interpolate.LinearNDInterpolator(
            avg[:, :2], avg[:, 2])
        # import pdb
        # pdb.set_trace()
        z_new = self.interp_func(x, y)
        no_nans = np.isnan(z_new) == False
        self.pts = self.pts[no_nans]
        self.polar = self.polar[no_nans]
        self.vals = self.vals[no_nans]
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

    def __init__(self, filetypes=ftypes,title='Select the images you want to process.'):
        self.app = app
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

    def __init__(self, filetypes=ftypes, title='Select a folder'):
        self.app = app
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

class tracker_window():

    def __init__(self, dirname="./"):
        # m.pyplot.ion()
        self.dirname = dirname
        self.load_filenames()
        self.num_frames = len(self.filenames)
        self.range_frames = np.array(range(self.num_frames))
        self.curr_frame_index = 0
        self.data_changed = False
        # the figure
        self.load_image()
        # figsize = self.image.shape[1]/90, self.image.shape[0]/90
        h, w = self.image.shape[:2]
        if w > h:
            fig_width = 8
            fig_height = h/w * fig_width
        else:
            fig_height = 8
            fig_width = w/h * fig_height
        # start with vmin and vmax at extremes
        self.vmin = 0
        self.vmax = np.iinfo(self.image.dtype).max
        self.vmax_possible = self.vmax
        # self.figure = plt.figure(1, figsize=(
        #     figsize[0]+1, figsize[1]+2), dpi=90)
        self.figure = plt.figure(1, figsize=(fig_width, fig_height), dpi=90)
        # xmarg, ymarg = .2, .1
        # fig_left, fig_bottom, fig_width, fig_height = .15, .1, .75, .85
        fig_left, fig_bottom, fig_width, fig_height = .1, .1, .75, .8
        axim = plt.axes([fig_left, fig_bottom, fig_width, fig_height])
        self.implot = plt.imshow(self.image, cmap='viridis', vmin=self.vmin, vmax=self.vmax)
        self.xlim = self.figure.axes[0].get_xlim()
        self.ylim = self.figure.axes[0].get_ylim()
        self.axis = self.figure.get_axes()[0]
        self.figure.axes[0].set_xlim(*self.xlim)
        self.figure.axes[0].set_ylim(*self.ylim)
        self.image_data = self.axis.images[0]
        # title
        self.title = self.figure.suptitle(
            '%d - %s' % (self.curr_frame_index + 1, self.filenames[self.curr_frame_index].rsplit('/')[-1]))

        # the slider controlling frames
        axframe = plt.axes([fig_left, 0.04, fig_width, 0.02])
        self.curr_frame = Slider(
            axframe, 'frame', 1, self.num_frames, valinit=1, valfmt='%d', color='k')
        self.curr_frame.on_changed(self.change_frame)
        # the vmin slider
        vminframe = plt.axes([fig_left + fig_width + .02, 0.1, .02, .05 + .7])
        self.vmin = Slider(
            vminframe, 'min', 0, self.vmax_possible,
            valinit=0, valfmt='%d', color='k', orientation='vertical')
        self.vmin.on_changed(self.show_image)
        # the vmax slider
        vmaxframe = plt.axes([fig_left + fig_width + .1, 0.1, .02, .05 + .7])
        self.vmax = Slider(
            vmaxframe, 'max', 0, self.vmax_possible, valinit=self.vmax_possible,
            valfmt='%d', color='k', orientation='vertical')
        self.vmax.on_changed(self.show_image)
        # limit both sliders
        self.vmin.slidermax = self.vmax
        self.vmax.slidermin = self.vmin
        # the colorbar in between
        self.cbar_ax = plt.axes([fig_left + fig_width + .06, 0.1, .02, .05 + .7])
        self.colorvals = np.arange(self.vmax_possible)
        self.cbar = self.cbar_ax.pcolormesh([0, 10],
                                            self.colorvals,
                                            self.colorvals[:, np.newaxis],
                                            cmap='viridis', vmin=0, vmax=self.vmax_possible)
        self.cbar_ax.set_xticks([])
        self.cbar_ax.set_yticks([])
        # connect some keys
        # self.cidk = self.figure.canvas.mpl_connect(
        #     'key_release_event', self.on_key_release)
        # self.cidm = self.figure.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        # self.cidm = self.figure.canvas.mpl_connect('', self.on_mouse_release)
        # self.figure.canvas.toolbar.home = self.show_image

        # change the toolbar functions
        NavigationToolbar2.home = self.show_image
        NavigationToolbar2.save = self.save_data

    def load_filenames(self):
        ls = os.listdir(self.dirname)
        self.filenames = [os.path.join(self.dirname, f) for f in ls if f.lower().endswith(
            ('.png', '.jpg', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff'))]
        self.filenames.sort()

    def load_image(self):
        print(self.curr_frame_index)
        self.image = PIL.Image.open(self.filenames[self.curr_frame_index])
        self.image = np.asarray(self.image)

    def show_image(self, *args):
        print('show_image')
        # first plotthe image
        self.im = np.copy(self.image)
        colorvals = np.copy(self.colorvals)
        # remove values > vmax
        self.im[self.im > self.vmax.val] = 0
        self.figure.axes[0].get_images()[0].set_clim([self.vmin.val, self.vmax.val])
        self.figure.axes[0].get_images()[0].set_data(self.im)
        colorvals[colorvals > self.vmax.val] = 0
        self.cbar.set_array(colorvals)
        self.cbar.set_clim([self.vmin.val, self.vmax.val])
        # and the title
        self.title.set_text('%d - %s' % (self.curr_frame_index + 1,
                                         self.filenames[self.curr_frame_index].rsplit('/')[-1]))
        plt.draw()

    def change_frame(self, new_frame):
        print('change_frame {} {}'.format(new_frame, int(new_frame)))
        self.curr_frame_index = int(new_frame)-1
        self.load_image()
        self.show_image()
        if self.data_changed:
            self.save_data()
            self.data_changed = False

    def nudge(self, direction):
        self.show_image()
        # self.change_frame(mod(self.curr_frame, self.num_frames))
        self.data_changed = True

    def on_key_release(self, event):
        # frame change
        if event.key in ("pageup", "alt+v", "alt+tab"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index, self.num_frames))
        elif event.key in ("pagedown", "alt+c", "tab"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 2, self.num_frames))
            print(self.curr_frame_index)
        elif event.key == "alt+pageup":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index - 9, self.num_frames))
        elif event.key == "alt+pagedown":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 11, self.num_frames))
        elif event.key == "home":
            self.curr_frame.set_val(1)
        elif event.key == "end":
            self.curr_frame.set_val(self.num_frames)
        # marker move
        elif event.key == "left":
            self.nudge(-1)
        elif event.key == "right":
            self.nudge(1)
        elif event.key == "up":
            self.nudge(-1j)
        elif event.key == "down":
            self.nudge(1j)
        elif event.key == "alt+left":
            self.nudge(-10)
        elif event.key == "alt+right":
            self.nudge(10)
        elif event.key == "alt+up":
            self.nudge(-10j)
        elif event.key == "alt+down":
            self.nudge(10j)

    def update_sliders(self, val):
        self.show_image()

    def on_mouse_release(self, event):
        self.change_frame(0)

    def save_data(self):
        print('save')
        for fn, val in zip(self.objects_to_save.keys(), self.objects_to_save.values()):
            np.save(fn, val)

class stackFilter():
    """Import image filenames filter images using upper and lower contrast bounds."""

    def __init__(self, fns=os.listdir("./")):
        """Import images using fns, a list of filenames."""
        self.fns = fns
        self.folder = os.path.dirname(self.fns[0])
        self.vals = None
        self.imgs = None

    def load_images(self):
        print("Loading images:\n")
        first_img = None
        for fn in self.fns:
            try:
                first_img = load_image(fn)
                break
            except:
                pass
        width, height = first_img.shape
        imgs = []
        for num, fn in enumerate(self.fns):
            try:
                imgs += [load_image(fn)]
            except:
                print(f"{fn} failed to load.")
                imgs += [np.zeros((width, height))]
            print_progress(num, len(self.fns))
        self.imgs = np.array(imgs, dtype=first_img.dtype)

    def contrast_filter(self):
        self.contrast_filter_UI = tracker_window(dirname=self.folder)
        plt.show()
        # grab low and high bounds from UI
        self.low = int(np.round(self.contrast_filter_UI.vmin.val))
        self.high = int(np.round(self.contrast_filter_UI.vmax.val))
        self.load_images()
        print("Extracting coordinate data: ")
        inds_to_remove = np.logical_or(self.imgs <= self.low, self.imgs > self.high)
        self.imgs[inds_to_remove] = 0
        # np.logical_and(self.imgs <= self.high, self.imgs > self.low, out=self.imgs)
        # self.imgs = self.imgs.astype(bool, copy=False)
        try:
            ys, xs, zs = np.where(self.imgs > 0)
            vals = self.imgs[ys, xs, zs]
        except:
            print(
                "coordinate data is too large. Using a hard drive memory map instead of RAM.")
            self.imgs_memmap = np.memmap(os.path.join(self.folder, "volume.npy"),
                                         mode='w+', shape=self.imgs.shape, dtype=bool)
            self.imgs_memmap[:] = self.imgs[:]
            del self.imgs
            self.imgs = None
            xs, ys, zs, vals = [], [], [], []
            for depth, img in enumerate(self.imgs_memmap):
                y, x = np.where(img > 0)
                z = np.repeat(depth, len(x))
                vals += [img[y, x]]
                xs = np.append(xs, x)
                ys = np.append(ys, y)
                zs = np.append(zs, z)
                print_progress(depth + 1, len(self.imgs))
        self.arr = np.array([xs, ys, zs], dtype=np.uint16).T
        self.vals = np.array(vals)

    def pre_filter(self):
        self.contrast_filter_UI = tracker_window(dirname=self.folder)
        plt.show()
        # grab low and high bounds from UI
        self.low = self.contrast_filter_UI.vmin.val
        self.high = self.contrast_filter_UI.vmax.val
        folder = os.path.join(self.folder, 'prefiltered_stack')
        if not os.path.isdir(folder):
            os.mkdir(folder)
        self.load_images()
        inds_to_remove = np.logical_or(self.imgs <= self.low, self.imgs > self.high)
        self.imgs[inds_to_remove] = 0
        # self.imgs = self.imgs.astype('uint8', copy=False)
        # np.multiply(self.imgs, 255, out=self.imgs)
        print("Saving filtered images:\n")
        for num, (fn, img) in enumerate(zip(self.fns, self.imgs)):
            base = os.path.basename(fn)
            new_fn = os.path.join(folder, base)
            save_image(new_fn, img)
            print_progress(num + 1, len(self.fns))


class ScatterPlot3d():
    """Plot 3d datapoints using pyqtgraph's GLScatterPlotItem."""

    def __init__(self, arr, color=None, size=1, window=None,
                 colorvals=None, cmap=plt.cm.viridis):
        self.arr = arr
        self.app = app
        self.color = color
        self.cmap = cmap
        self.size = size
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
        self.window.show()
        self.app.exec_()


class ScatterPlot2d():
    """Plot 2d datapoints using pyqtgraph's GLScatterPlotItem."""

    # def __init__(self, arr, color=(1, 1, 1, 1), size=[1], app=None, window=None):
    def __init__(self, arr, color=None, window=None, size=1, axis=None, colorvals=None,
                 cmap=plt.cm.viridis, scatter_GUI=None):
        self.app = app
        self.arr = np.array(arr)
        self.color = color
        self.colorvals = colorvals
        self.cmap = cmap
        self.size = size
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
    eye = Points(eye, center_points=True, rotate_com=True, vals=SF.vals)
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
        elif self.filename.endswith(".pkl"):
            return pickle.load(open(self.filename, "rb"))


def main():
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
        "cross_section.points",
        # "ommatidia_centers.npy",
        "ommatidia_labels.npy",
        "ommatidia_measurements.csv",
        "inter_ommatidial_measurements.csv"]
    custom_names = [
        'Image stack was prefiltered.',
        '3D coordinates were processed.',
        'The cross section was extracted.',
        # 'Ommatidia centers were extracted.',
        'Ommatidia clusters were found.',
        'Ommatidial measurements were calculated.',
        'Inter-ommatidial measurements were calculated.']
    functions = [
        pre_filter,             # optional
        import_stack,           # works even if no prefilter
        get_cross_section,
        # find_ommatidia_centers,
        find_ommatidia_clusters,
        get_ommatidial_measurements,
        get_interommatidial_measurements,
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
    for num, mark in enumerate(benchmarks):
        if mark.exists():
            print(f"{num+1}. {mark.statement}")
    choice = None
    while choice not in np.arange(len(functions)).astype(str):
        choice = input(
            "Press the number to continue from that benchmark or press <0> to go to the main menu: ")
    choice = int(choice)
    progress = choice
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
    benchmark = benchmarks[max(progress - 1, 0)].load()
    print("Would you like to preview the outcome of each step? ")
    preview = None
    while preview not in ['0', '1']:
        preview = input("Press <1> for yes and <0> for no. ")
    preview = preview == '1'
    input_val = benchmark
    for function in functions:
        input_val = function(input_val, preview,
                             project_folder=project_folder)

img_filetypes = ['tiff', 'tff', 'tif', 'jpeg', 'jpg', 'png', 'bmp']

def pre_filter(home_folder):
    fns = os.listdir(home_folder)
    fns = [os.path.join(home_folder, fn) for fn in fns if fn.split(".")[-1].lower() in img_filetypes]
    fns.sort()
    assert len(fns) > 1, print(
        f"The folder {home_folder} has no image files.")
    SF = stackFilter(fns)
    SF.pre_filter()
    return home_folder


def import_stack(folder, preview=True, **kwargs):
    # return coordinates as a Points object
    if 'project_folder' in kwargs.keys():
        project_folder = kwargs['project_folder']
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
            if eye.vals is None or len(set(eye.vals)) == 1:
                scatter = ScatterPlot3d(eye.pts)
            else:
                scatter = ScatterPlot3d(eye.pts, colorvals=eye.vals)
            scatter.show()
        print("Save and continue? ")
        response = None
        while response not in ['0', '1']:
            response = input("Press <1> for yes or <0> to load "
                             "and filter the images again? ")
        save = response == '1'
    eye.save(os.path.join(home_folder, "compound_eye_data", "coordinates.points"))
    return eye

def get_cross_section(eye, preview=True, thickness=.3, **kwargs):
    if 'project_folder' in kwargs.keys():
        project_folder = kwargs['project_folder']
    else:
        project_folder = os.path.dirname(eye.latest_fn)
    save = False
    while save is False:
        eye.get_polar_cross_section(thickness=thickness)
        cross_section = eye.cross_section
        # this part is very slow. can we perform it on just a subset of the points?
        if preview:
            # print("Rasterizing the cross section data: ")
            # img = cross_section.rasterize(image_size=10**6)
            # pg.image(img)
            # plt.imshow(img)
            # plt.show()
            if cross_section.vals is None or len(set(cross_section.vals)) == 1:
                scatter = ScatterPlot3d(cross_section.pts)
            else:
                scatter = ScatterPlot3d(cross_section.pts, colorvals=cross_section.vals)
            scatter.show()
        response = None
        print("Save and continue? ")
        while response not in ['0', '1']:
            response = input("Press <1> for yes or <0> to extract "
                             "the cross section using a different thickness?")
        if response == "1":
            save = True
        else:
            thickness = np.inf
            while not np.logical_and(thickness >= 0, thickness <= 1):
                thickness = input("What proportion of thickness (from 0 to 1) should "
                                  "we use?")
                try:
                    thickness = float(thickness)
                except:
                    thickness = np.inf
    eye.save(eye.latest_fn)
    # save_image(os.path.join(project_folder, "cross_section_raster.png"),
    #            img.astype('uint8'))
    cross_section.save(os.path.join(project_folder, "cross_section.points"))
    return cross_section


def find_pairs(set1, set2, max_iters=3, chunk_size=1000):
    set1_no_nans = np.any(np.isnan(set1), axis=1) == False
    set2_no_nans = np.any(np.isnan(set2), axis=1) == False
    set1 = np.copy(spherical_to_cartesian(set1[set1_no_nans]))
    set2 = np.copy(spherical_to_cartesian(set2[set2_no_nans]))
    tree = spatial.KDTree(set1)
    dists, inds = tree.query(set2)
    inds = inds.astype(float)
    vals, counts = np.unique(inds, return_counts=True)
    duplicates = counts > 1
    unassigned_set1_inds = np.array(sorted(
        set(np.arange(set1.shape[0])) - set(inds.astype(int))))
    iters = 0
    while np.any(duplicates) and len(unassigned_set1_inds) > 0 and iters < max_iters:
        nans = np.isnan(inds)
        # create a smaller tree with just the points in set1 have no pair
        tree = spatial.KDTree(set1[unassigned_set1_inds])
        # from that tree, query the points nearest the points in set2
        # with no pair, indicated by their nan index value
        sub_dists, sub_inds = tree.query(set2[nans])
        # replace global index list with new pairs
        inds[nans] = unassigned_set1_inds[sub_inds]
        # update distance list
        dists[nans] = sub_dists
        # are there duplicates?
        vals, counts = np.unique(inds, return_counts=True)
        duplicates = counts > 1
        # what is left unassigned?
        unassigned_set1_inds = np.array(
            sorted(set(np.arange(set1.shape[0])) - set(inds.astype(int))))
        # replace duplicates that have dist > minimum distance with nan
        for val in vals[duplicates]:
            sub_dists = dists[inds == val]
            sub_inds = np.where(inds == val)[0][np.argsort(sub_dists)[1:]]
            inds[sub_inds] = np.nan
        iters += 1
    no_nans = np.isnan(inds) == False
    set2 = set2[no_nans]
    inds = inds[no_nans]
    dists = dists[no_nans]
    set1 = set1[inds.astype(int)]
    return cartesian_to_spherical(set1), cartesian_to_spherical(set2), dists


def find_nearest_neighbors(pts, centers, pad=.1):
    thetas, phis, radii = pts.T
    centers_thetas, centers_phis, centers_radii = centers.T
    theta_range, phi_range = thetas.max() - thetas.min(), phis.max() - phis.min()
    theta_num_segments = int(np.ceil(theta_range/(np.pi / 4)))
    phi_num_segments = int(np.ceil(phi_range/(np.pi / 4)))
    theta_boundaries = np.linspace(thetas.min(), thetas.max(), theta_num_segments + 1)
    phi_boundaries = np.linspace(phis.min(), phis.max(), phi_num_segments + 1)
    theta_length, phi_length = np.diff(theta_boundaries)[0], np.diff(phi_boundaries)[0]
    theta_centers = (theta_boundaries + theta_length/2)[:-1]
    phi_centers = (phi_boundaries + phi_length/2)[:-1]
    theta_pad, phi_pad = pad * theta_length, pad * phi_length
    all_inds = np.empty(len(pts), dtype=float)
    all_inds.fill(np.nan)
    all_dists = np.copy(all_inds)
    num = 0
    for theta_ind, (theta_min, theta_center, theta_max) in enumerate(zip(
            theta_boundaries[:-1], theta_centers, theta_boundaries[1:])):
        for phi_ind, (phi_min, phi_center, phi_max) in enumerate(zip(
                phi_boundaries[:-1], phi_centers, phi_boundaries[1:])):
            # find pts within interval
            pts_inds = ((thetas >= theta_min) *
                        (thetas < theta_max) *
                        (phis >= phi_min) *
                        (phis < phi_max))
            pts_near = pts[pts_inds]
            # find centers within interval
            centers_inds = ((centers_thetas >= theta_min - theta_pad) *
                            (centers_thetas < theta_max + theta_pad) *
                            (centers_phis >= phi_min - phi_pad) *
                            (centers_phis < phi_max + phi_pad))
            centers_near = centers[centers_inds]
            # convert to rectangular coordinates to fix spherical distortion
            pts_near_rect = spherical_to_cartesian(np.copy(pts_near))
            centers_near_rect = spherical_to_cartesian(np.copy(centers_near))
            # remove centerpoints with nans
            no_nans = np.isnan(centers_near_rect) == False
            centers_near_rect = centers_near_rect[np.any(no_nans, axis=1)]
            # find desired radius (ideally, will contain neighbors)
            centers_tree = spatial.KDTree(centers_near_rect)
            neighbor_dists, neighbor_inds = centers_tree.query(centers_near_rect, k=2)
            radius = 1.5 * np.median(neighbor_dists[:, 1])
            pts_tree = spatial.KDTree(pts_near_rect)  # use to find points near center
            for center in centers_near_rect:
                # find ball of pts_near_rect within radius of center
                nearby_pts_inds = np.array(pts_tree.query_ball_point(center, r=radius))
                if len(nearby_pts_inds) > 3:
                    nearby_pts = pts_near_rect[nearby_pts_inds]
                    nearby_centers_inds = np.array(centers_tree.query_ball_point(center, r=radius))
                    nearby_centers = centers_near_rect[nearby_centers_inds]
                    new_spherical = cartesian_to_spherical(nearby_pts - center)
                    # use spectral clustering to find clusters in ball
                    clusterer = cluster.SpectralClustering(n_clusters=len(nearby_centers),
                        assign_labels='discretize', random_state=0).fit(nearby_pts)
                    labels = clusterer.labels_
                    # find cluster nearest the center point
                    cluster_means = []
                    labels_set = sorted(set(labels))
                    for lbl in labels_set:
                        cluster_means += [nearby_pts[labels == lbl].mean(0)]
                    cluster_means = np.array(cluster_means)
                    cluster_dists = np.linalg.norm(cluster_means - center, axis=1)
                    center_label = labels_set[np.argmin(cluster_dists)]
                    # get index for pts in cluster
                    pts_center_inds = np.where(pts_inds)[0][nearby_pts_inds][labels == center_label]
                    all_inds[pts_center_inds] = num
                    all_dists[pts_center_inds] = np.linalg.norm(nearby_pts[labels == center_label] - center, axis=1)
                    num += 1
                    print_progress(num, len(centers))
                    # troubleshooting:
                    # scatter = ScatterPlot3d(nearby_pts - center, size=5, colorvals=labels)
                    # scatter_centers = ScatterPlot3d(nearby_centers - center,
                    #                                 window=scatter.window,
                    #                                 size=10, color=(1, 0, 0, 1))
                    # scatter.show()
            # centers_tree = spatial.KDTree(centers_near_rect)
            # dists_near, inds_near = centers_tree.query(pts_near_rect)
            # all_inds[pts_inds] = np.where(centers_inds)[0][inds_near]
            # all_dists[pts_inds] = dists_near
            print_progress(theta_ind * phi_num_segments + phi_ind + 1,
                           theta_num_segments * phi_num_segments)
    return all_inds, all_dists

# troubleshooting find_pairs
# set1 = np.random.random((30000, 3))
# set1 += 1
# set2 = np.copy(set1)
# noise = np.random.normal(0, .0001, set2.shape)
# set2 = set2 + noise
# set2 = set2[3000:]
# res1, res2 = find_pairs(set1, set2)
# ts = np.arange(len(res1))
# np.random.shuffle(ts)
# plt.scatter(res1.T[0], res1.T[1], c = ts)
# plt.scatter(res2.T[0], res2.T[1], c = ts)
# plt.show()
# troubleshooting find_nearest_neighbors
# if 'app' not in locals():
#     app = QApplication([])
# set1 = np.random.random((30000, 3))
# set1 
# set1 -= .5
# set1[:, 2] = 1
# set2_inds = np.random.choice(np.arange(set1.shape[0]), 1000000)
# set2 = set1[set2_inds]
# noise = np.random.normal(0, .001, set2.shape)
# set2 += noise
# fitted_inds, fitted_dists = find_nearest_neighbors(set2, set1, .1)
# no_nans = np.isnan(fitted_inds) == False
# scatter = ScatterPlot3d(set2[no_nans], size=1, colorvals=fitted_inds[no_nans], app=app)
# scatter2 = ScatterPlot3d(set1, size=3, colorvals=np.arange(len(set1)), window=scatter.window, app=app)
# scatter2.show()
# plt.scatter(set2[:, 0], set2[:, 1], c=fitted_inds, marker='.')
# plt.scatter(set1[:, 0], set1[:, 1], marker='o', color='k')
# plt.show() 





def find_ommatidia_clusters(cross_section, preview=True, **kwargs):
    save = False
    # pixel_length = .001
    image_size = 10**6
    image_size = 350000
    # find pixel size for a given image size
    if 'project_folder' in kwargs.keys():
        project_folder = kwargs['project_folder']
    project_folder = os.path.dirname(cross_section.latest_fn)
    if 'app' in kwargs.keys():
        app = kwargs['app']
    eye = load_Points(os.path.join(project_folder, "coordinates.points"))
    # segment the cross section into polar squares of size pi x pi
    # find range of theta and phi
    theta_range = cross_section.theta.max() - cross_section.theta.min()  # max = pi
    phi_range = cross_section.phi.max() - cross_section.phi.min()        # max = 2pi
    theta_num_segments = int(np.ceil(2 * theta_range/np.pi))
    phi_num_segments = int(np.ceil(2 * phi_range/np.pi))
    theta_boundaries = np.linspace(cross_section.theta.min(), 
                                 cross_section.theta.max(),
                                 theta_num_segments + 1)
    phi_boundaries = np.linspace(cross_section.phi.min(), 
                                 cross_section.phi.max(),
                                 phi_num_segments + 1)
    theta_length, phi_length = np.diff(theta_boundaries)[0], np.diff(phi_boundaries)[0]
    theta_centers = (theta_boundaries + theta_length/2)[:-1]
    phi_centers = (phi_boundaries + phi_length/2)[:-1]
    theta_pad, phi_pad = .1 * theta_length, .1 * phi_length
    # go through each segment and process ommatidia centers
    inner_shell = eye.residuals < 0
    outer_shell = eye.residuals >= 0
    labels = []
    dist_trees = []
    # minimum_facets = 500
    # maximum_facets = 50000
    minimum_facets = 2000
    maximum_facets = 8000
    # the shell method doesn't work for ommatidia that are very askew
    # for skew ommatidia, let's try an older method:
    # get the center points of ommatidia using the cross section from the previous step
    # make distance tree of the whole eye
    # for each center, 
        # find all points within generous radius of the center
        # rotate a cylinder (radius ~ .5 * avg. distance to neighbors, height = 1.1 * thickness)
        # settle on the cylinder orientation with the most points contained.
        # label these points with the cluster label

    save = False
    while not save:
        thetas = []
        phis = []
        segment_image_size = image_size / (theta_num_segments * phi_num_segments)
        print("Processing segments of the polar coordinates: ")
        for theta_ind, (theta_min, theta_center, theta_max) in enumerate(zip(
                theta_boundaries[:-1], theta_centers, theta_boundaries[1:])):
            for phi_ind, (phi_min, phi_center, phi_max) in enumerate(zip(
                    phi_boundaries[:-1], phi_centers, phi_boundaries[1:])):
                cont = False
                while not cont:
                    inds = ((cross_section.theta > theta_min - theta_pad) *
                            (cross_section.theta < theta_max + theta_pad) *
                            (cross_section.phi > phi_min - phi_pad) *
                            (cross_section.phi < phi_max + phi_pad))
                    cross_section_segment = cross_section.pts[inds]
                    phi_displacement = phi_center - np.pi
                    theta_displacement = theta_center - np.pi/2
                    rot = rotate(cross_section_segment, phi_displacement, axis=2).T
                    rot = rotate(rot, theta_displacement, axis=1).T
                    polar1 = cartesian_to_spherical(rot)
                    segment = Points(rot, rotate_com=False, polar=polar1,
                                     spherical_conversion=False,
                                     vals=cross_section.vals[inds])
                    # 3. use low pass filter method from Eye object in fly_eye to find
                    # centers of cone clusters/ommatidia.
                    # a. rasterize the image so that we can use our image processing
                    # algorithm in fly_eye
                    img, (theta_vals, phi_vals) = segment.rasterize(image_size=segment_image_size)
                    mask = img > 0
                    mask = convex_hull_image(mask)
                    cross_section_eye = fe.Eye(img, pixel_size=segment.raster_pixel_length,
                                               mask=mask)
                    cross_section_eye.theta_vals, cross_section_eye.phi_vals = theta_vals, phi_vals
                    cross_section_eye.get_ommatidia(
                        max_facets=maximum_facets / (theta_num_segments * phi_num_segments),
                        min_facets=minimum_facets / (theta_num_segments * phi_num_segments),
                        method=1, method1_sigma=2)
                    if cross_section_eye.ommatidia is not None:
                        segment_phis, segment_thetas = cross_section_eye.ommatidia
                        segment_thetas += theta_vals.min()
                        segment_phis += phi_vals.min()
                        segment_polar = np.array([segment_thetas, segment_phis, np.ones(len(segment_phis))]).T
                        segment_pts = spherical_to_cartesian(segment_polar)
                        rot = rotate(segment_pts, -theta_displacement, axis=1).T
                        rot = rotate(rot, -phi_displacement, axis=2).T
                        polar1 = cartesian_to_spherical(rot)
                        theta, phi, _ = polar1.T
                        within_bounds = ((theta >= theta_min) *
                                         (theta < theta_max) *
                                         (phi >= phi_min) *
                                         (phi < phi_max))
                        thetas += [theta[within_bounds]]
                        phis += [phi[within_bounds]]
                        if preview:
                            plt.pcolormesh(theta_vals, phi_vals, img.T)
                            plt.scatter(segment_thetas, segment_phis, color='r', marker='.')
                            plt.gca().set_aspect('equal')
                            plt.tight_layout()
                            plt.xlabel("polar angle (theta)")
                            plt.ylabel("azimuthal angle (phi)")
                            plt.title("Eye Segment with Ommatidial Centers")
                            plt.show()
                        print("Continue with the same minimum and maximum ommatidial counts? ")
                        response = None
                        while response not in ['0', '1']:
                            response = input("Press <1> for yes or <0> to analyze "
                                             "the image using a different limits: ")
                        cont = response == '1'
                    else:
                        print("Failed to extract ommatidia. Choose new ommatidia count limits: ")
                        cont = False
                    if not cont:
                        minimum_facets = input("What is the minimum number of ommatidia? ")
                        success = False
                        while success is False:
                            try:
                                minimum_facets = int(minimum_facets)
                                success = True
                            except:
                                minimum_facets = input("The response must be a whole number: ")
                        maximum_facets = input("What is the maximum number of ommatidia? ")
                        success = False
                        while success is False:
                            try:
                                maximum_facets = int(maximum_facets)
                                success = True
                            except:
                                maximum_facets = input("The response must be a whole number: ")
                    print_progress(theta_ind * phi_num_segments + phi_ind + 1,
                                   theta_num_segments * phi_num_segments)
        print("\n")
        thetas = np.concatenate(thetas)
        phis = np.concatenate(phis)
        cross_section.fit_surface()
        radii = cross_section.interp_func(thetas, phis)
        radii = eye.interp_func(thetas, phis)
        img, (theta_vals, phi_vals) = cross_section.rasterize(image_size=image_size)
        if preview:
            plt.pcolormesh(theta_vals, phi_vals, img.T)
            plt.scatter(thetas, phis, color='r', marker='.')
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.xlabel("polar angle (theta)")
            plt.ylabel("azimuthal angle (phi)")
            plt.show()
        centers = np.array([thetas, phis, radii]).T
        print("Continue with this image size? ")
        response = None
        while response not in ['0', '1']:
            response = input("Press <1> for yes or <0> to rasterize "
                             "the image using a different image size: ")
        save = response == '1'
        if not save:
            image_size = input("What image size should we use (pixel count)? ")
            success = False
            while success is False:
                try:
                    image_size = int(image_size)
                    success = True
                except:
                    image_size = input("The response must be a whole number: ")
        else:
            dist_trees += [centers]
    all_inds, all_dists = find_nearest_neighbors(eye.polar, centers)
    # for cross_section in [eye[inner_shell], eye[outer_shell]]:
    #     save = False
    #     while not save:
    #         thetas = []
    #         phis = []
    #         segment_image_size = image_size / (theta_num_segments * phi_num_segments)
    #         print("Processing segments of the polar coordinates: ")
    #         for theta_ind, (theta_min, theta_center, theta_max) in enumerate(zip(
    #                 theta_boundaries[:-1], theta_centers, theta_boundaries[1:])):
    #             for phi_ind, (phi_min, phi_center, phi_max) in enumerate(zip(
    #                     phi_boundaries[:-1], phi_centers, phi_boundaries[1:])):
    #                 cont = False
    #                 while not cont:
    #                     inds = ((cross_section.theta > theta_min - theta_pad) *
    #                             (cross_section.theta < theta_max + theta_pad) *
    #                             (cross_section.phi > phi_min - phi_pad) *
    #                             (cross_section.phi < phi_max + phi_pad))
    #                     cross_section_segment = cross_section.pts[inds]
    #                     phi_displacement = phi_center - np.pi
    #                     theta_displacement = theta_center - np.pi/2
    #                     rot = rotate(cross_section_segment, phi_displacement, axis=2).T
    #                     rot = rotate(rot, theta_displacement, axis=1).T
    #                     polar1 = cartesian_to_spherical(rot)
    #                     segment = Points(rot, rotate_com=False, polar=polar1,
    #                                      spherical_conversion=False,
    #                                      vals=cross_section.vals[inds])
    #                     # 3. use low pass filter method from Eye object in fly_eye to find
    #                     # centers of cone clusters/ommatidia.
    #                     # a. rasterize the image so that we can use our image processing
    #                     # algorithm in fly_eye
    #                     img, (theta_vals, phi_vals) = segment.rasterize(image_size=segment_image_size)
    #                     mask = img > 0
    #                     mask = convex_hull_image(mask)
    #                     cross_section_eye = fe.Eye(img, pixel_size=segment.raster_pixel_length,
    #                                                mask=mask)
    #                     cross_section_eye.theta_vals, cross_section_eye.phi_vals = theta_vals, phi_vals
    #                     cross_section_eye.get_ommatidia(
    #                         max_facets=maximum_facets / (theta_num_segments * phi_num_segments),
    #                         min_facets=minimum_facets / (theta_num_segments * phi_num_segments),
    #                         method=1)
    #                     if cross_section_eye.ommatidia is not None:
    #                         segment_phis, segment_thetas = cross_section_eye.ommatidia
    #                         segment_thetas += theta_vals.min()
    #                         segment_phis += phi_vals.min()
    #                         segment_polar = np.array([segment_thetas, segment_phis, np.ones(len(segment_phis))]).T
    #                         segment_pts = spherical_to_cartesian(segment_polar)
    #                         rot = rotate(segment_pts, -theta_displacement, axis=1).T
    #                         rot = rotate(rot, -phi_displacement, axis=2).T
    #                         polar1 = cartesian_to_spherical(rot)
    #                         theta, phi, _ = polar1.T
    #                         within_bounds = ((theta >= theta_min) *
    #                                          (theta < theta_max) *
    #                                          (phi >= phi_min) *
    #                                          (phi < phi_max))
    #                         thetas += [theta[within_bounds]]
    #                         phis += [phi[within_bounds]]
    #                         if preview:
    #                             plt.pcolormesh(theta_vals, phi_vals, img.T)
    #                             plt.scatter(segment_thetas, segment_phis, color='r', marker='.')
    #                             plt.gca().set_aspect('equal')
    #                             plt.tight_layout()
    #                             plt.xlabel("polar angle (theta)")
    #                             plt.ylabel("azimuthal angle (phi)")
    #                             plt.title("Eye Segment with Ommatidial Centers")
    #                             plt.show()
    #                         print("Continue with the same minimum and maximum ommatidial counts? ")
    #                         response = None
    #                         while response not in ['0', '1']:
    #                             response = input("Press <1> for yes or <0> to analyze "
    #                                              "the image using a different limits: ")
    #                         cont = response == '1'
    #                     else:
    #                         print("Failed to extract ommatidia. Choose new ommatidia count limits: ")
    #                         cont = False
    #                     if not cont:
    #                         minimum_facets = input("What is the minimum number of ommatidia? ")
    #                         success = False
    #                         while success is False:
    #                             try:
    #                                 minimum_facets = int(minimum_facets)
    #                                 success = True
    #                             except:
    #                                 minimum_facets = input("The response must be a whole number: ")
    #                         maximum_facets = input("What is the maximum number of ommatidia? ")
    #                         success = False
    #                         while success is False:
    #                             try:
    #                                 maximum_facets = int(maximum_facets)
    #                                 success = True
    #                             except:
    #                                 maximum_facets = input("The response must be a whole number: ")
    #                     print_progress(theta_ind * phi_num_segments + phi_ind + 1,
    #                                    theta_num_segments * phi_num_segments)
    #         print("\n")
    #         thetas = np.concatenate(thetas)
    #         phis = np.concatenate(phis)
    #         cross_section.fit_surface()
    #         radii = cross_section.interp_func(thetas, phis)
    #         img, (theta_vals, phi_vals) = cross_section.rasterize(image_size=image_size)
    #         if preview:
    #             plt.pcolormesh(theta_vals, phi_vals, img.T)
    #             plt.scatter(thetas, phis, color='r', marker='.')
    #             plt.gca().set_aspect('equal')
    #             plt.tight_layout()
    #             plt.xlabel("polar angle (theta)")
    #             plt.ylabel("azimuthal angle (phi)")
    #             plt.show()
    #         centers = np.array([thetas, phis, radii]).T
    #         # tree = spatial.KDTree(centers)
    #         # find clusters here and now! by finding point nearest each center
    #         # dists, inds = [], []
    #         # # this takes a lot of time and RAM so break it into chunks
    #         # chunks = int(np.round(cross_section.polar.shape[0] / 10**4))
    #         # chunks = np.array_split(cross_section.polar[:, :2], chunks, axis=0)
    #         # old_prop = 0
    #         # for num, coords in enumerate(chunks):
    #         #     d, i = tree.query(coords, k=1)
    #         #     dists += [d]
    #         #     inds += [i]
    #         #     prop = int(100 * ((num + 1) / len(chunks)))
    #         #     print_progress(num + 1, len(chunks))
    #         # print(".\n")
    #         # inds = np.concatenate(inds)
    #         # if preview:
    #         #     # get random colorvals for scatterplot
    #         #     cvals = {}
    #         #     rand_vals = np.random.choice(list(set(inds)), len(set(inds)), replace=False)
    #         #     for val, rand_val in zip(set(inds), rand_vals):
    #         #         cvals[val] = rand_val
    #         #     color_vals = np.array([cvals[ind] for ind in  inds])
    #         #     polar_scatter = ScatterPlot3d(cross_section.pts, app=app,
    #         #                                   colorvals=color_vals, cmap=plt.cm.tab20)
    #         #     polar_scatter.show()
    #         # segment_labels = inds
    #         print("Continue with this image size? ")
    #         response = None
    #         while response not in ['0', '1']:
    #             response = input("Press <1> for yes or <0> to rasterize "
    #                              "the image using a different image size: ")
    #         save = response == '1'
    #         if not save:
    #             image_size = input("What image size should we use (pixel count)? ")
    #             success = False
    #             while success is False:
    #                 try:
    #                     image_size = int(image_size)
    #                     success = True
    #                 except:
    #                     image_size = input("The response must be a whole number: ")
    #         else:
    #             dist_trees += [centers]
    #             # dist_trees += [tree]
    #             # labels += [segment_labels]
    # # inner_tree, outer_tree = dist_trees
    # # inner_pts, outer_pts = inner_tree.data, outer_tree.data
    # inner_pts, outer_pts = dist_trees
    # inner_pts, outer_pts, dists = find_pairs(inner_pts, outer_pts)
    # pairs = np.array([np.arange(len(inner_pts)), np.arange(len(inner_pts))]).T.astype(int)
    # max_dist = 1.5 * np.median(dists)
    # if preview:
    #     for (t0_ind, t1_ind), dist in zip(pairs, dists):
    #         if dist < max_dist:
    #             xs, ys = np.array([inner_pts[t0_ind, :2], outer_pts[t1_ind, :2]]).T
    #             plt.plot(xs, ys, 'k-')
    #             plt.scatter(xs, ys, c=[0, 1], marker='.', cmap='viridis')
    #     plt.gca().set_aspect('equal')
    #     plt.show()
    # breakpoint()
    # pairs = pairs[dists <= max_dist]
    # # pairs will allow us to convert from inner cluster labels to outer cluster labels
    # # now the inside and outside labels are the same
    # print("Finding clusters of points near ommatidia centers of inside shell: ")
    # thickness = np.diff(np.percentile(eye.residuals, [.5, 99.5]))[0]
    # # why is this not working at all? troubleshoot find_nearest_neighbors
    # inner_lbls, inner_dists = find_nearest_neighbors(eye[inner_shell].polar, inner_pts, .1)
    # print("\n")
    # print("Finding clusters of points near ommatidia centers of outside shell: ")
    # outer_lbls, outer_dists = find_nearest_neighbors(eye[outer_shell].polar, outer_pts, .1)
    # print("\n")
    # # inner_lbls, outer_lbls = labels
    # # new_inner_lbls = inds[inner_lbls]
    # all_inds = np.zeros(outer_shell.shape[0])
    # all_inds[inner_shell] = inner_lbls
    # all_inds[outer_shell] = outer_lbls

    # find all labels with less than 10 pts and assign as nan
    breakpoint()
    labels, counts = np.unique(all_inds, return_counts=True)
    labels_to_remove = labels[counts < 5]
    remove_inds = np.array([lbl in labels_to_remove for lbl in all_inds])
    all_inds[remove_inds] = np.nan
    no_nans = np.isnan(all_inds) == False  # remove nans from both labels and coordinate data
    eye = eye[no_nans]
    all_inds = all_inds[no_nans]
    if preview:
        # get random colorvals for scatterplot
        rand_vals = np.random.permutation(np.arange(0, all_inds.max() + 1)).astype(int)
        colorvals = rand_vals[all_inds.astype(int)]
        polar_scatter = ScatterPlot3d(eye.pts, colorvals=colorvals, cmap=plt.cm.tab20)
        polar_scatter.show()
    eye.save(os.path.join(project_folder, "coordinates.points"))
    np.save(os.path.join(project_folder, "ommatidia_labels.npy"), all_inds)
    return all_inds


def get_ommatidial_measurements(cluster_labels, preview=True, **kwargs):
    if 'project_folder' in kwargs.keys():
        project_folder = kwargs['project_folder']
    else:
        project_folder = os.getcwd()
    cluster_labels = np.array(cluster_labels)
    eye = load_Points(os.path.join(project_folder, "coordinates.points"))
    data_to_save = dict()
    # cols = ['x_center', 'y_center', 'z_center', 'theta_center',
    #         'phi_center', 'r_center', 'children_rectangular', 'children_polar', 'n']
    cols = ['x_center', 'y_center', 'z_center',
            'theta_center', 'phi_center', 'r_center', 'n',
            'aspect_ratio']
    for col in cols:
        data_to_save[col] = []
    # for num, cone in enumerate(clusters):
    cone_centers = []
    labels, counts = np.unique(cluster_labels, return_counts=True)
    print("Taking preliminary measurements per ommatidial cluster: ")
    for num, lbl in enumerate(labels):
        inds = cluster_labels == lbl
        cone = eye[inds]
        x_center, y_center, z_center = cone.pts.astype(float).mean(0)
        cone_centers += [[x_center, y_center, z_center]]
        theta_center, phi_center, r_center = cone.polar.astype(
            float).mean(0)
        children_pts, children_polar = cone.pts.astype(
            float), cone.polar.astype(float)
        n = len(children_pts)
        svd = np.linalg.svd(cone.pts.astype(float))[1]
        aspect_ratio = svd[0] / svd[1]
        for lbl, vals in zip(
                cols,
                # [x_center, y_center, z_center, theta_center, phi_center, r_center,
                #  children_pts.tolist(), children_polar.tolist(), n]):
                [x_center, y_center, z_center, theta_center, phi_center,
                 r_center, n, aspect_ratio]):
            data_to_save[lbl] += [vals]
        print_progress(num, len(labels))
    print("\n")
    cone_cluster_data = pd.DataFrame.from_dict(data_to_save)
    cone_cluster_data.to_csv(os.path.join(project_folder, "ommatidia_measurements.csv"))
    cone_centers = np.array(cone_centers)
    ns = cone_cluster_data['n']
    no_nans = np.isnan(ns) == False
    scatter_ns = ScatterPlot3d(
        cone_centers[no_nans],
        size=10,
        colorvals=ns[no_nans])
    scatter_ns.show()
    aspect_ratios = cone_cluster_data.aspect_ratio
    no_nans = np.isnan(aspect_ratios) == False
    scatter_aspect_ratio = ScatterPlot3d(
        cone_centers[no_nans],
        size=10,
        colorvals=aspect_ratios[no_nans])
    scatter_aspect_ratio.show()
    no_nans = np.isnan(cone_centers).sum(1) == 0
    tree = spatial.KDTree(cone_centers[no_nans])
    dists, inds = tree.query(cone_centers[no_nans], k=13)
    dists = dists[:, 1:]
    upper_limit = np.percentile(dists.flatten(), 99)
    dists = dists[dists < upper_limit].flatten() 
    clusterer = cluster.KMeans(2, init=np.array(
        [0, 100]).reshape(-1, 1), n_init=1)
    groups = clusterer.fit_predict(dists[:, np.newaxis])
    upper_limit = dists[groups == 0].max()
    # find lower and upper bounds for 'ommatidial diameters' (using set of all distances)
    # 5. Using our set of cone clusters, and the curvature implied by nearest cones,
    # we can take measurements relevant to the eye's optics.
    save = False
    while save is False:
        nearest_neighbors = spatial.KDTree(cone_centers)
        # dists, lbls = nearest_neighbors.query(cone_centers, k=13)
        neighbor_dists, neighbor_lbls = nearest_neighbors.query(
            cone_centers, k=7)
        neighbor_dists = neighbor_dists[:, 1:]
        neighbor_dists[neighbor_dists > upper_limit] = np.nan
        neighbor_lbls = neighbor_lbls[:, 1:]
        big_neighborhood_dists, big_neighborhood_lbls = nearest_neighbors.query(
            cone_centers, k=51)
        # grab points adjacent to the center point by:
        # 1. grab 12 points nearest the center point
        # 2. cluster the points into 2 groups, near and far
        # 3. filter out the far group
        lens_area = []
        anatomical_vectors = []
        approx_vectors = []
        skewness = []
        # cones = []
        print("Taking measurements per ommatidial cluster: ")
        for num, (center, cone_lbl, neighbor_group, dists, big_group) in enumerate(zip(
                cone_centers, labels, neighbor_lbls, neighbor_dists,
                big_neighborhood_lbls)):
            cone = eye[cluster_labels == cone_lbl]
            neighborhood = cone_centers[neighbor_group]
            big_neighborhood = cone_centers[big_group]
            # cone.neighbor_lbls = labels[neighbor_group]
            # approximate lens diameter using distance to nearest neighbors
            diam = np.nanmean(dists)
            area = np.pi * (.5 * diam) ** 2
            lens_area += [area]
            # cone.lens_area = area
            # approximate ommatidial axis vector by referring to the center of a
            # sphere fitting around the nearest neighbors
            pts = Points(big_neighborhood, center_points=False)
            pts.spherical()
            d_vector = pts.center - center
            d_vector /= LA.norm(d_vector)
            # instead of sphere fitting, use average normal vector of plane created by each pair of neighboring ommatidia_centers
            pts = cone_centers[neighbor_group]
            # center about the main ommatidium
            pts_centered = big_neighborhood - center
            pts_centered = pts_centered[1:]
            # find cross product of each pair of neighboring ommatidia
            cross = np.cross(pts_centered[np.newaxis], pts_centered[:, np.newaxis])
            # half of these will be left handed while the other half are right handed
            # because the matrix is negative symmetrical
            # find those that minimize the angle between itself and the vector to the
            # center of the fitted sphere
            magn = np.linalg.norm(cross, axis=-1)
            # normalize the direction vectors into unit vectors
            cross /= magn[:, :, np.newaxis]
            # find angle between cross product unit vectors and sphere center unit vector 
            sphere_vector = -center
            sphere_vector /= np.linalg.norm(sphere_vector)
            angles_between = np.arccos(np.dot(cross, sphere_vector))
            avg_vector = cross[angles_between < np.pi/2]
            d_vector = avg_vector.mean(0)
            # center
            approx_vectors += [d_vector]
            # cone.approx_vector = d_vector
            # approximate ommatidial axis vector by regressing cone data
            d_vector2 = cone.get_line()
            d_vector2 = np.array([d_vector2, -d_vector2])
            angs = np.arccos(np.dot(d_vector2, sphere_vector))
            ind = angs == angs.min()
            d_vector2 = d_vector2[ind].min(0)
            anatomical_vectors += [d_vector2]
            # cone.anatomical_vector = d_vector2
            # plot sample cone with through lines (check)
            # ptsa = (np.array([-20, 20]) * d_vector[:, np.newaxis]).T
            # ptsb = (np.array([-20, 20]) * d_vector2[:, np.newaxis]).T
            # scatter = ScatterPlot3d(cone.pts - cone.pts.mean(0), app=app)
            # plt1 = gl.GLLinePlotItem(pos=ptsa, color=(1, 0, 0, 1),
            #                          width=5, antialias=True)
            # scatter.window.addItem(plt1)
            # plt2 = gl.GLLinePlotItem(pos=ptsb, color=(0, 1, 0, 1),
            #                         width=5, antialias=True)
            # scatter.window.addItem(plt2)
            # scatter.show()
            # calculate the anatomical skewness (angle between the two vectors)
            # inside_ang = min(
            #     angle_between(d_vector, d_vector2),
            #     angle_between(d_vector, -d_vector2))
            inside_ang = angle_between(d_vector, d_vector2)
            skewness += [inside_ang]
            # cone.skewness = inside_ang
            # cones += [cone]
            print_progress(num + 1, len(cone_centers))
        print("\n")
        lens_area = np.array(lens_area)
        anatomical_vectors = np.array(anatomical_vectors)
        approx_vectors = np.array(approx_vectors)
        skewness = np.array(skewness)
        labels = np.array(labels)
        # neighbor_inds = np.array(neighbor_lbls)  # indeces of neighboring clusters
        neighbor_lbls = labels[neighbor_lbls]  # label referring to ommatidia_labels.npy
        # TODO: 3 scatter plots showing the three parameters
        no_nans = np.isnan(lens_area) == False
        scatter_lens_area = ScatterPlot3d(
            cone_centers[no_nans],
            size=10,
            colorvals=lens_area[no_nans])
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
    cone_cluster_data['label'] = labels
    # cone_cluster_data['neighbor_inds'] = neighbor_inds.tolist()  # index in this dataframe
    cone_cluster_data['neighbor_labels'] = neighbor_lbls.tolist()  # refers to ommatidia_labels.npy
    cone_cluster_data.to_csv(os.path.join(project_folder, "ommatidia_measurements.csv"))
    cone_cluster_data.to_pickle(os.path.join(project_folder, "ommatidia_measurements.pkl"))
    # with open(os.path.join(project_folder, "ommatidia_clusters.pkl"), 'wb') as pickle_file:
    #     pickle.dump(clusters, pickle_file)
    return cone_cluster_data


def get_interommatidial_measurements(cone_cluster_data, preview=True, **kwargs):
    if 'project_folder' in kwargs.keys():
        project_folder = kwargs['project_folder']
    else:
        project_folder = os.getcwd()
    labels = cone_cluster_data.neighbor_labels.values
    pairs_tested = set()
    interommatidial_angle_approx = dict()
    interommatidial_angle_anatomical = dict()
    orientations_dict = dict()
    eye = load_Points(os.path.join(project_folder, "coordinates.points"))
    cluster_lbls = np.load(os.path.join(project_folder, "ommatidia_labels.npy"))
    approx_IOAs = []
    anatomical_IOAs = []
    pairs = []
    orientations = []
    radii = []
    intersections = []
    aspect_ratios = []
    # for neighbor in cone.neighbor_lbls:
    for num, cone in cone_cluster_data.iterrows():
        if isinstance(cone.neighbor_labels, str):
            neighbor_labels = cone.neighbor_labels[1:-1].split(",")
            neighbor_labels = [int(float(val)) for val in neighbor_labels]
        else:
            neighbor_labels = cone.neighbor_labels
        # store the centerpoint between each cone
        if isinstance(cone.approx_axis, str):
            approx_direction = np.array(cone.approx_axis[1:-1].split(",")).astype(float)
        else:
            approx_direction = np.array(cone.approx_axis)
        pts = eye.pts[cluster_lbls == cone.label]
        lbl = cone.label
        for neighbor_ind in neighbor_labels:
            pair = tuple(sorted([lbl, neighbor_ind]))
            if pair not in pairs:
                i = np.where(cone_cluster_data.label.values == neighbor_ind)[0][0]
                neighbor_cone = cone_cluster_data.loc[i]
                neighbor_pts = eye.pts[cluster_lbls == neighbor_ind]
                # let's get the anatomical IOA
                # first, find the best fitting plane to the union of the two clusters
                all_pts = np.append(pts, neighbor_pts, axis=0)
                # using singular value decomposition, we can find the two principle components
                # of all_pts, then project each cluster onto this basis
                principle_components = np.linalg.svd(all_pts - all_pts.mean(0))[2]
                inv_principle_components = np.linalg.inv(principle_components)
                pts_proj = np.dot(pts - all_pts.mean(0), principle_components)
                pts_proj[:, 2] = 0
                neighbor_pts_proj = np.dot(neighbor_pts - all_pts.mean(0), principle_components)
                neighbor_pts_proj[:, 2] = 0
                # now, get the direction vectors for each projection using the first
                # pricinple component of each projection
                pts_svd = np.linalg.svd(pts_proj - pts_proj.mean(0))
                pts_direction = pts_svd[2][0]
                neighbor_pts_svd = np.linalg.svd(
                    neighbor_pts_proj - neighbor_pts_proj.mean(0))
                neighbor_pts_direction = neighbor_pts_svd[2][0]
                aspect_ratio = min(pts_svd[1][0] / pts_svd[1][1],
                                   neighbor_pts_svd[1][0] / neighbor_pts_svd[1][1])
                # now, the anatomical IOA is the angle between these direction vectors
                anatomical_IOA = angle_between(pts_direction, neighbor_pts_direction)
                anatomical_IOA = min(anatomical_IOA, np.pi - anatomical_IOA)
                # find the intersection point using these two direction vectors in the original basis
                pts_direction_original = np.dot(
                    pts_direction, inv_principle_components)
                neighbor_pts_direction_original = np.dot(
                    neighbor_pts_direction, inv_principle_components)
                dir1 = pts_direction_original
                dir2 = neighbor_pts_direction_original
                pos1 = pts.mean(0)
                pos2 = neighbor_pts.mean(0)
                inter1 = intersection(pos1, dir1, pos2, dir2)
                inter2 = intersection(pos2, dir2, pos1, dir1)
                inter = (inter1 + inter2)/2
                sidea = np.linalg.norm(pos1 - inter)
                sideb = np.linalg.norm(pos2 - inter)
                sidec = np.linalg.norm(pos2 - pos1)
                radius = (sidea + sideb)/2
                # check that the intersection is inside the sphere, not outside
                # find distance from the center
                inter_radius = np.linalg.norm(inter)
                if inter_radius > cone.r_center:
                    anatomical_IOA = -anatomical_IOA
                    radius = -radius
                ang = np.arccos((sidea**2 + sideb**2 - sidec**2)/(2 * sidea * sideb))
                # troubleshooting: ang should be very close to anatomical_IOA
                if abs(anatomical_IOA) > np.pi/2:
                    scatter = ScatterPlot3d(pts - all_pts.mean(0), color=(1, 1, 0, 1))
                    scatter2 = ScatterPlot3d(neighbor_pts - all_pts.mean(0),
                                             color=(1, 0, 1, 1),
                                             window=scatter.window)
                    scatter.show()
                    pts_center = pts_proj.mean(0)[:2]
                    pts_offset = 10 * pts_direction[:2]
                    pts_xs, pts_ys = np.array(
                        [pts_center - pts_offset,
                         pts_center + pts_offset]).T
                    neighbor_center = neighbor_pts_proj.mean(0)[:2]
                    neighbor_offset = 10 * neighbor_pts_direction[:2]
                    neighbor_xs, neighbor_ys = np.array(
                        [neighbor_center - neighbor_offset,
                         neighbor_center + neighbor_offset]).T
                    plt.scatter(pts_proj[:, 0], pts_proj[:, 1], color='r')
                    plt.scatter(neighbor_pts_proj[:, 0], neighbor_pts_proj[:, 1], color='g')
                    plt.plot(pts_xs, pts_ys, color='r')
                    plt.plot(neighbor_xs, neighbor_ys, color='g')
                    plt.gca().set_aspect('equal')
                    plt.show()
                    breakpoint()
                # now, the approx IOA:
                # get the approx direction vector from the neighboring cone
                if isinstance(neighbor_cone.approx_axis, str):
                    neighbor_approx_direction = np.array(
                        neighbor_cone.approx_axis[1:-1].split(",")).astype(float)
                else:
                    neighbor_approx_direction = np.array(neighbor_cone.approx_axis)
                # get angle between these vectors, accounting for potential skewness
                # approx_IOA = angle_between_skew_vectors(
                #     pts.mean(0), approx_direction,
                #     neighbor_pts.mean(0), neighbor_approx_direction
                #     )
                all_vectors = np.append(
                    approx_direction[np.newaxis],
                    neighbor_approx_direction[np.newaxis], axis=0)
                principle_components = np.linalg.svd(all_vectors)[2]
                approx_direction_proj = np.dot(approx_direction, principle_components)[:2]
                neighbor_approx_direction_proj = np.dot(
                    neighbor_approx_direction, principle_components)[:2]
                approx_IOA = angle_between(
                    approx_direction_proj, neighbor_approx_direction_proj)
                # now, measure the orientation in reference to the polar angles
                th1, ph1 = cone.theta_center, cone.phi_center
                th2, ph2 = neighbor_cone.theta_center, neighbor_cone.phi_center
                orientation = np.arctan2(ph2 - ph1, th2 - th1)
                orientations += [orientation]
                radii += [radius]
                intersections += [inter]
                aspect_ratios += [aspect_ratio]
                anatomical_IOAs += [anatomical_IOA]
                approx_IOAs += [approx_IOA]
                pairs += [pair]
        print_progress(num + 1, len(labels))
    print("\n")
    pairs_tested = np.array(list(pairs))
    IOA_approx = np.array(list(approx_IOAs))
    IOA_anatomical = np.array(list(anatomical_IOAs))
    orientations = np.array(list(orientations))
    radii = np.array(list(radii))
    intersections = np.array(list(intersections))
    # breakpoint()
    # scatter = ScatterPlot3d(intersections, colorvals=orientations)
    # point = ScatterPlot3d(intersections.mean(0)[np.newaxis], window=scatter.window, size=10)
    # scatter.show()
    data_to_save = dict()
    cols = ['cluster1', 'cluster2',
            'cluster1_x', 'cluster1_y', 'cluster1_z',
            'cluster2_x', 'cluster2_y', 'cluster2_z', 
            'cluster1_theta', 'cluster1_phi', 'cluster1_radii',
            'cluster2_theta', 'cluster2_phi', 'cluster2_radii',
            'approx_angle', 'anatomical_angle', 'orientation', 'radius']
    for col in cols:
        data_to_save[col] = []
    for num, (pair, approx, anatomical, orientation, radius) in enumerate(
            zip(pairs_tested, IOA_approx, IOA_anatomical, orientations, radii)):
        ind1, ind2 = pair
        ind1 = np.where(cone_cluster_data.label.values == ind1)[0][0]
        ind2 = np.where(cone_cluster_data.label.values == ind2)[0][0]
        cluster1 = cone_cluster_data.loc[ind1]
        cluster2 = cone_cluster_data.loc[ind2]
        for lbl, vals in zip(
                cols,
                [ind1, ind2,
                 cluster1.x_center, cluster1.y_center, cluster1.z_center,
                 cluster2.x_center, cluster2.y_center, cluster2.z_center, 
                 cluster1.theta_center, cluster1.phi_center, cluster1.r_center,
                 cluster2.theta_center, cluster2.phi_center, cluster2.r_center,
                 approx, anatomical, orientation, radius]):
            data_to_save[lbl] += [vals]
        print_progress(num, len(pairs_tested))
    print("\n")
    cone_pair_data = pd.DataFrame.from_dict(data_to_save)
    cone_pair_data.to_csv(os.path.join(project_folder, "interommatidial_measurements.csv"))
    return

def close():
    return

if __name__ == "__main__":
    main()
