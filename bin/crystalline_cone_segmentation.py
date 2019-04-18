# from matplotlib import pyplot as plt
# from matplotlib import colors, cm
# from scipy import spatial, interpolate
# import numpy as np
# from numpy import linalg as LA
# import hdbscan
# import math
# import pickle
# import pandas as pd

from compound_eye_tools import *

# def bootstrap_ci(arr, reps=1000, ci_range=[2.5, 97.5], stat_func=np.mean):
#     pseudo_distro = np.random.choice(arr, (len(arr), reps))
#     if stat_func is not None:
#         pseudo_distro = stat_func(pseudo_distro, axis=1)
#     l, h = np.percentile(pseudo_distro, ci_range, axis=0)
#     return l, h

# def rotate(arr, theta, axis=0):
#     if axis == 0:
#         rot_matrix = np.array(
#             [[1, 0, 0],
#              [0, np.cos(theta), -np.sin(theta)],
#              [0, np.sin(theta), np.cos(theta)]])
#     elif axis == 1:
#         rot_matrix = np.array(
#             [[np.cos(theta), 0, np.sin(theta)],
#              [0, 1, 0],
#              [-np.sin(theta), 0, np.cos(theta)]])
#     elif axis == 2:
#         rot_matrix = np.array(
#             [[np.cos(theta), -np.sin(theta), 0],
#              [np.sin(theta), np.cos(theta), 0],
#              [0, 0, 1]])
#     return np.dot(arr, rot_matrix)

# def sphereFit(spX, spY, spZ):
#     #   Assemble the f matrix
#     f = np.zeros((len(spX), 1))
#     f[:, 0] = (spX**2) + (spY**2) + (spZ**2)
#     A = np.zeros((len(spX), 4))
#     A[:, 0] = spX*2
#     A[:, 1] = spY*2
#     A[:, 2] = spZ*2
#     A[:, 3] = 1
#     C, residules, rank, sigval = np.linalg.lstsq(A, f, rcond=None)
#     #   solve for the radius
#     t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
#     radius = math.sqrt(t)
#     return radius, np.squeeze(C[:-1])

# def cartesian_to_spherical(pts, center=np.array([0, 0, 0])):
#     pts = pts - center
#     radii = LA.norm(np.copy(pts), axis=1)
#     theta = np.arccos(np.copy(pts)[:, 0]/radii)
#     phi = np.arctan2(np.copy(pts)[:, 1], np.copy(pts)[:, 2]) + np.pi
#     polar = np.array([theta, phi, radii])
#     return polar.T

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

# 1. convert to spherical coorinates by fitting a sphere with OLS
eye = np.load("./eye_only.npy").astype(float)
eye = Points(eye)
eye.spherical()

# 2. fit a surface to the spherical data, modelling each r as a function of theta and phi.
th_min, th_max = np.percentile(eye.theta, [.5, 99.5])
ph_min, ph_max = np.percentile(eye.phi, [.5, 99.5])

thetas, phis = np.linspace(th_min, th_max, 10), np.linspace(ph_min, ph_max, 10)
knotst, knotsp = thetas.copy(), phis.copy()
knotst[0] += .0001
knotst[-1] -= .0001
knotsp[0] += .0001
knotsp[-1] -= .0001

lut = interpolate.LSQSphereBivariateSpline(
    eye.theta, eye.phi, eye.radii,
    knotst, knotsp)

thetas, phis = np.linspace(0, np.pi, 100), np.linspace(0, 2*np.pi, 100)
t = np.pi/2.
mod_y = lut(t, phis)
mod_p = phis
t_i = np.logical_and(eye.theta < t + .1, eye.theta > t - .1)
data_y = eye.radii[t_i]
data_p = eye.phi[t_i]

# plt.scatter(data_p, data_y, marker='.')
# plt.plot(mod_p, mod_y[0], 'r-')
# plt.show()

# 3. extract a sheet of points around the approximate surface and use HDBSCAN to segment the crystalline cone centers
surface = lut(eye.theta, eye.phi, grid=False)
residuals = eye.radii - surface
sheet_thickness = np.percentile(abs(residuals), 10)
low, high = surface - sheet_thickness, surface + sheet_thickness
sheet_inds = np.logical_and(eye.radii <= high, eye.radii >= low)
sheet = eye[sheet_inds]

np.array([eye.theta[sheet_inds], eye.phi[sheet_inds]]).T

clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
labels = clusterer.fit_predict(sheet.pts)

new_labels = np.array(sorted(set(np.copy(labels))))
np.random.shuffle(new_labels)
conv_labels = dict()
for lbl, new_lbl in zip(sorted(set(labels)), new_labels):
    conv_labels[lbl] = new_lbl
new_labels = np.array([conv_labels[lbl] for lbl in labels])

vals = new_labels.copy()
vals = vals + vals.min()
vals = vals / vals.max()
c = cm.viridis(vals)
c[labels < 0, -1] = 0
# plt.scatter(sheet.theta, sheet.phi, color=c)

coord_centers = []
polar_centers = []
for lbl in sorted(set(labels)):
    i = labels == lbl
    c = sheet.pts[i].mean(0)
    p = sheet.polar[i].mean(0)
    coord_centers.append(c)
    polar_centers.append(p)

coord_centers = np.array(coord_centers)
polar_centers = np.array(polar_centers)
np.save("./coord_centers.npy", coord_centers)
np.save("./polar_centers.npy", polar_centers)

centers = Points(coord_centers, polar=polar_centers, center_points=False)
# plt.scatter(centers.theta, centers.phi, color='k', marker='o')
# ax = plt.gca()
# ax.set_aspect('equal')

# 4. convert cone centers back to cartesian coordinates. For each center, find the nearest cluster of points within a generous radius.
safe_radius = np.percentile(residuals, 99)
neighbors_tree = spatial.KDTree(eye.pts)
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, algorithm='boruvka_kdtree')
cones = []
for center in coord_centers:
    i = neighbors_tree.query_ball_point(center, r=safe_radius)
    near_pts = eye.pts[i]
    near_polar = eye.polar[i]
    if len(near_pts) >= 10:
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
            cones += [Points(near_pts[lbl], polar=near_polar[lbl], center_points=False)]

# 4b. store data to a spreadsheet for measurements in the next step
cols = ['x_center','y_center','z_center','theta_center','phi_center','r_center',
        'children_pts','children_polar','n']
data_to_save = dict()
for col in cols:
    data_to_save[col] = []
for cone in cones:
    cone.spherical()
    x_center, y_center, z_center = cone.pts.astype(float).mean(0)
    theta_center, phi_center, r_center = cone.polar.astype(float).mean(0)
    children_pts, children_polar = cone.pts.astype(float), cone.polar.astype(float)
    n = len(children_pts)
    for lbl, vals in zip(
            cols,
            [x_center, y_center, z_center, theta_center, phi_center, r_center,
             children_pts, children_polar, n]):
        data_to_save[lbl] += [vals]

cone_cluster_data = pd.DataFrame.from_dict(data_to_save)
cone_cluster_data.to_csv("./cone_cluster_data.csv")

cone_centers = [cone.pts for cone in cones]
with open("./cone_clusters.pkl", "wb") as fn:
    pickle.dump(cones, fn)

# 5. Using our set of cone clusters, and the curvature of the thin sheet, we can take measurements relevant to the eye's optics.



