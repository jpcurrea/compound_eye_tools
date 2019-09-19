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
from PyQt5.QtWidgets import QWidget, QFileDialog

def load_image(fn):
    return np.asarray(PIL.Image.open(fn))

def print_progress(part, whole, bar=True):
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
    m = data.mean(0)
    max_val = np.round(2*abs(data - m).max()).astype(int)
    uu, dd, vv = np.linalg.svd(data - m)
    return vv[component]

def bootstrap_ci(arr, reps=1000, ci_range=[2.5, 97.5], stat_func=np.mean):
    pseudo_distro = np.random.choice(arr, (len(arr), reps))
    if stat_func is not None:
        pseudo_distro = stat_func(pseudo_distro, axis=1)
    l, h = np.percentile(pseudo_distro, ci_range, axis=0)
    return l, h

def rotate(arr, theta, axis=0):
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
    #   Assemble the f matrix
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
    # theta must be between 
    # TODO: center points based on center of mass
    # TODO: fix boundary problems for trig functions
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
    It is setup for indexing and is used throughout the pipeline.
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
                rot1 = rotate(self.pts.T, ang1, axis=0)
                # 3. rotate com along z axis (com[2]) until y (com[1]) = 0
                ang2 = np.arctan2(com1[1], com1[0])
                rot2 = rotate(rot1.T, ang2, axis=2)
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
    ('tiff images', '*.tiff *.TIFF *.tff *.TFF'),
    ('bmp images', '*.bmp *.BMP'),
    ('all types', '*.*')]

ftypes = [f"{fname} ({ftype})" for (fname, ftype) in filetypes]
ftypes = ";;".join(ftypes)

class fileSelector(QWidget):
    """Offers a file selection dialog box with filters based on common image filetypes.
    """
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
