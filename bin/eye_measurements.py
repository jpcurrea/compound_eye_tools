# from matplotlib import pyplot as plt
# from matplotlib import colors, cm
# from scipy import spatial, interpolate
# import numpy as np
# from numpy import linalg as LA
# import hdbscan
# import math
# import pickle
# import pandas as pd

# class Points():
#     def __init__(self, arr, center_points=True, polar=None):
#         self.pts = np.array(arr)
#         if arr.ndim > 1:
#             assert self.pts.shape[1] == 3, "Input array should have shape N x 3. Instead it has shape {} x {}.".format(self.pts.shape[0], self.pts.shape[1])
#         else:
#             assert self.pts.shape[0] == 3, "Input array should have shape 3 or N x 3. Instead it has shape {}.".format(self.pts.shape)
#             self.pts = self.pts.reshape((1, -1))
#         self.shape = self.pts.shape
#         self.center = np.array([0, 0, 0])
#         self.polar = None
#         if polar is not None:
#             self.polar = polar
#             self.theta, self.phi, self.radii = self.polar.T
#         if center_points:
#             # fit sphere
#             x, y, z = self.pts.T
#             self.radius, self.center = sphereFit(x, y, z)
#             # center points using the center of that sphere
#             self.pts = self.pts -  self.center
#             self.center = self.center - self.center
#             # rotate points using the center of mass in theta and phi
#             com = self.pts.mean(0)
#             polar = cartesian_to_spherical(com.reshape((1, -1)))
#             self.polar_com = polar
#             theta_offset, phi_offset, r_m = np.squeeze(polar)
#             theta_offset = np.pi/2. - theta_offset
#             phi_offset = np.pi - phi_offset
#             self.pts = rotate(self.pts, theta_offset, axis=1)
#             self.pts = rotate(self.pts, theta_offset, axis=2)
#             # self.pts = rotate(self.pts, phi_offset, axis=0)
#         # grab spherical coordinates of centered points
#         self.x, self.y, self.z = self.pts.T

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, key):
#         out = Points(self.pts[key], polar=self.polar[key], center_points=False)
#         return out

#     def spherical(self, center=None):
#         if center is None:
#             center = self.center
#         self.polar = cartesian_to_spherical(self.pts, center=center)
#         self.theta, self.phi, self.radii = self.polar.T

from compound_eye_tools import *

fn = "./cone_clusters.pkl"
with open(fn, 'rb') as cone_data:
    cones = pickle.load(cone_data)

cone_centers = np.array([cone.pts.mean(0) for cone in cones])
# df = pd.read_csv("./cone_cluster_data.csv")
# cone_centers = np.array([df.x_center, df.y_center, df.z_center]).T

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 1000
w.show()
w.setWindowTitle('Moth Eye Ommatidia')

scatter = gl.GLScatterPlotItem(pos=cone_centers, size=5, color=(0,0,1,1))
w.addItem(scatter)
