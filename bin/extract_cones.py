from compound_eye_tools import *
from pyqtgraph.Qt import QtCore, QtGui
from sklearn import cluster
import pyqtgraph.opengl as gl
import pyqtgraph as pg
app = QtGui.QApplication([])
pqt_window = gl.GLViewWidget()
pqt_window.opts['distance'] = 1000
pqt_window.show()
pqt_window.setWindowTitle('Moth Eye Ommatidia')
 
print("0. Select the range of densities predominantly in the crystalline cones, filter the data, and save.")

# 0a. let user select, from images at different orientations,
# the range of densities corresponding to the crystalline cones
folder = "./"
fns = os.listdir(folder)
fns = [os.path.join(folder, fn) for fn in fns if fn.endswith(".tif")]
fns = sorted(fns)

zs = len(fns)
imgs = []
print("Loading images:\n")
for num, fn in enumerate(fns):
    imgs += [load_image(fn)]
    print_progress(num, zs)
    
arr = np.array(imgs, dtype=np.uint16)
img = imgs[int(.5*zs)]

slide = pg.image(arr)
slide.setPredefinedGradient('greyclip')

ret = input("Use the bottom slide bar or the arrow keys to change slides and the right slide bar to select the range of density values corresponding predominantly to the crystaline cones. Left click and drag with the mouse to move the image around and use the scroll wheel to zoom in and out. Press <Enter> once you are happy with the filer.")

slide.close()
del imgs

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

print("0b. Segment the eye of interest using HDBSCAN.")
# 0b. segment the eye using a density filter and HDBSCAN
head = arr.astype(np.int16)
del arr
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
scatter = gl.GLScatterPlotItem(pos=head[selection], size=1)
pqt_window.addItem(scatter)

ret = input("HDBSCAN clusters a random subsection (1%) to get macro structures.")
pqt_window.removeItem(scatter)

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
        pqt_window.addItem(scatter)
        options[num + 1] = lbl
        print(fg[txt_color] + "{}.\t{}".format(str(num + 1), len(group)) + fg.rs)


choice = int(input("Type the number corresponding to the eye cluster: "))
choice_lbl = options[choice]

for scatter in scatters:
    pqt_window.removeItem(scatter)

scatter = gl.GLScatterPlotItem(pos=clusters[choice_lbl], size=1)
pqt_window.addItem(scatter)

choice = input("Displaying sample eye data only. To find other points within cluster, press <Enter>.")
pqt_window.removeItem(scatter)
cluster_labels, cluster_strengths = [], []
print("Finding other points in cluster:")
for i in np.arange(0, len(head), 10000):
    labels, strengths = hdbscan.approximate_predict(clusterer, head[i:i+10000])
    cluster_labels += [labels]
    cluster_strengths += [strengths]
    print_progress(i, len(head))
cluster_labels = np.concatenate(cluster_labels)
eye = head[cluster_labels == choice_lbl]

np.save("./eye_only.npy", eye)
del head

scatter = gl.GLScatterPlotItem(pos=eye, size=1)
pqt_window.addItem(scatter)

choice = input("Crystalline cone data saved. Press <Enter> to close.")
pqt_window.removeItem(scatter)

print("1 - 4. Convert to spherical coordinates, fit surface to the data, approximate crystaline cone centers, and finally segment the crystaline cone clusters.")
# 1. convert to spherical coorinates by fitting a sphere with OLS
eye = np.load("./eye_only.npy")
eye = Points(eye.astype(float))
eye.spherical()

# 2. fit a surface to the spherical data, modelling each r as a function of theta and phi.
thetas = np.linspace(eye.theta.min(), eye.theta.max(), 100)
phis = np.linspace(eye.phi.min(), eye.phi.max(), 100)

avg = []
for col_num, (t1, t2) in enumerate(zip(thetas[:-1], thetas[1:])):
    col = []
    in_column = np.logical_and(eye.theta >= t1, eye.theta < t2)
    in_column = eye.polar[in_column]
    for row_num, (p1, p2) in enumerate(zip(phis[:-1], phis[1:])):
        in_row = np.logical_and(in_column[:, 1] >= p1, in_column[:, 1] < p2)
        if any(in_row):
            avg += [np.median(in_column[in_row], axis=0)]
    print_progress(col_num, len(thetas) - 1)
avg = np.array(avg)

# filter outlier points by using bootstraped 95% confidence band (not of the mean)
l, h = bootstrap_ci(avg[:, 2], reps=1000, ci_range=[2.5, 97.5], stat_func=None)
l, h = l.mean(), h.mean()
avg = avg[np.logical_and(avg[:, 2] >= l, avg[:, 2] < h)]

t, p, r = avg.T
tck = interpolate.bisplrep(t, p, r, s=3)

# 3. extract a sheet of points around the approximate surface and use HDBSCAN to segment the crystalline cone centers
# surface = lut(eye.theta, eye.phi, grid=False)
surface = []
for th, ph in zip(eye.theta, eye.phi):
    surface += [interpolate.bisplev(th, ph, tck)]
surface = np.array(surface)

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
c = plt.cm.viridis(vals)
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
for num, center in enumerate(coord_centers):
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
    print_progress(num, len(coord_centers))

# 4b. store data to a spreadsheet for measurements in the next step
cols = ['x_center','y_center','z_center','theta_center','phi_center','r_center',
        'children_pts','children_polar','n']
data_to_save = dict()
for col in cols:
    data_to_save[col] = []
for num, cone in enumerate(cones):
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
    print_progress(num, len(cones))

cone_cluster_data = pd.DataFrame.from_dict(data_to_save)
cone_cluster_data.to_csv("./cone_cluster_data.csv")

cone_centers = [cone.pts for cone in cones]
with open("./cone_clusters.pkl", "wb") as fn:
    pickle.dump(cones, fn)

cone_centers = np.array([cone.pts.mean(0) for cone in cones])

# 5. Using our set of cone clusters, and the curvature of the thin sheet, we can take measurements relevant to the eye's optics.
with open("./cone_clusters.pkl", "rb") as pkl_fn:
    cones = pickle.load(pkl_fn)


nearest_neighbors = spatial.KDTree(cone_centers)
dists, lbls = nearest_neighbors.query(cone_centers, k=13)

# grab points adjacent to the center point by:
# 1. grab 12 points nearest the center point
# 2. cluster the points into 2 groups, near and far
# 3. filter out the far group

lens_area = []                  # based on distance to nearest neighbors
anatomical_vectors = []            # based on line down cone axis
approx_vectors = []             # based on normal vector of interpolation
skewness = []                   # angle difference between ioas
neighbor_lbls = []              # index labels corresponding to neighboring points

clusterer = cluster.KMeans(2, init=np.array([0, 100]).reshape(-1, 1), n_init=1)
for lbl, (center, cone) in enumerate(zip(cone_centers, cones)):
    neighborhood = cone_centers[lbls[lbl]]
    neighbor_dists = dists[lbl].reshape(-1, 1)
    neighbor_groups = clusterer.fit_predict(neighbor_dists[1:])
    neighbor_lbls += [lbls[lbl][1:][neighbor_groups == 0]]
    cone.neighbor_lbls = lbls[lbl][1:][neighbor_groups == 0]
    
    # approximate lens diameter using distance to nearest neighbors
    diam = np.mean(neighbor_dists[1:][neighbor_groups == 0])
    area = np.pi * (.5 * diam) ** 2
    lens_area += [area]
    cone.lens_area = area

    # approximate ommatidial axis vector by referring to the center of a fitted sphere
    pts = Points(neighborhood, center_points=False)
    pts.spherical()
    d_vector = pts.center - center
    d_vector /= LA.norm(d_vector)
    approx_vectors += [d_vector]
    cone.approx_vector = d_vector
    
    # approximate ommatidial axis vector by regressing cone data
    d_vector2 = cone.get_line()
    anatomical_vectors += [d_vector2]
    cone.anatomical_vector = d_vector2

    # calculate the anatomical skewness (angle between the two vectors)
    inside_ang = min(
        angle_between(d_vector, d_vector2),
        angle_between(d_vector, -d_vector2))
    skewness += [inside_ang]
    cone.skewness = inside_ang

    print_progress(lbl, len(cone_centers))


lens_area = np.array(lens_area)
anatomical_vectors = np.array(anatomical_vectors)
approx_vectors = np.array(approx_vectors)
skewness = np.array(skewness)
neighbor_lbls = np.array(neighbor_lbls)

cone_cluster_data['lens_area'] = lens_area
cone_cluster_data['anatomical_axis'] = anatomical_vectors.tolist()
cone_cluster_data['approx_axis'] = approx_vectors.tolist()
cone_cluster_data['skewness'] = skewness
cone_cluster_data.to_csv("./cone_cluster_data.csv")

vals = skewness / skewness.max()
c = plt.cm.viridis(vals)
cone_centers = np.array([cone.pts.mean(0) for cone in cones])
scatter = gl.GLScatterPlotItem(pos=cone_centers, size=5, color=c)
pqt_window.addItem(scatter)

# go through each pair of adjacent cones, and find inside angle between both vector types
# i = 1000
# d_vector1 = approx_vectors[i]
# d_vector2 = anatomical_vectors[i]
# cone = cones[i]
# lbls = neighbor_lbls[i]

# for cone in cones:
#     # grab adjacent cones, determined earlier
#     neighbors = [cones[lbl] for lbl in cone.neighbor_lbls]
#     for neighbor in neighbors:
        


# scatter = gl.GLScatterPlotItem(pos=head[selection], size=1)
# pqt_window.addItem(scatter)

# neighbors = gl.GLScatterPlotItem(pos=pts.pts, size=5, color=(1,1,1,1))
# pqt_window.addItem(neighbors)

# md = gl.MeshData.sphere(rows=20, cols=20)
# m3 = gl.GLMeshItem(meshdata=md, drawFaces=False, drawEdges=True,
#                    edgeColor=(0, 0, 1, .3), smooth=True)#, shader='balloon')
# r, c = pts.radius, pts.center
# m3.scale(r, r, r)
# m3.translate(c[0], c[1], c[2])
# pqt_window.addItem(m3)

# phi_centers, radii = pts.theta, pts.radii - pts.radius

# spherical_plot = pg.plot(title="Spherical Projection")
# center_points = pg.ScatterPlotItem(
#     phi_centers, radii,
#     pen=None, symbol='o',
#     # pxMode=False,
#     size=10,
#     color=[1, 1, 1, 1])
# spherical_plot.addItem(center_points)



# cone_scatter = gl.GLScatterPlotItem(pos=cone.pts - cone.pts.mean(0), size=1, color=(1,0,0,1))
# pqt_window.addItem(cone_scatter)

# # m = pts.mean(0)
# m = 0
# a1, a2 = m - 10*d_vector, m - 10*d_vector2
# b1, b2 = m + 10*d_vector, m + 10*d_vector2

# l_graph = gl.GLLinePlotItem(pos=np.array([a1, b1]), color=[1,1,1,1])
# pqt_window.addItem(l_graph)

# l_graph2 = gl.GLLinePlotItem(pos=np.array([a2, b2]), color=[0,0,1,1])
# pqt_window.addItem(l_graph2)

app.exec_()
