from compound_eye_tools import *
import pandas as pd

data = pd.read_pickle("../../cluster_data/21968/stack/cone_cluster_data.pkl")

# go through each pair of adjacent cones, and find inside angle between both vector types
random_cone = data.iloc[1000]
d_vector1 = random_cone.
d_vector2 = anatomical_vectors[i]
cone = cones[i]
lbls = neighbor_lbls[i]

for cone in cones:
    # grab adjacent cones, determined earlier
    neighbors = [cones[lbl] for lbl in cone.neighbor_lbls]
    for neighbor in neighbors:
        


scatter = gl.GLScatterPlotItem(pos=head[selection], size=1)
pqt_window.addItem(scatter)

neighbors = gl.GLScatterPlotItem(pos=pts.pts, size=5, color=(1,1,1,1))
pqt_window.addItem(neighbors)

md = gl.MeshData.sphere(rows=20, cols=20)
m3 = gl.GLMeshItem(meshdata=md, drawFaces=False, drawEdges=True,
                   edgeColor=(0, 0, 1, .3), smooth=True)#, shader='balloon')
r, c = pts.radius, pts.center
m3.scale(r, r, r)
m3.translate(c[0], c[1], c[2])
pqt_window.addItem(m3)

phi_centers, radii = pts.theta, pts.radii - pts.radius

spherical_plot = pg.plot(title="Spherical Projection")
center_points = pg.ScatterPlotItem(
    phi_centers, radii,
    pen=None, symbol='o',
    # pxMode=False,
    size=10,
    color=[1, 1, 1, 1])
spherical_plot.addItem(center_points)



cone_scatter = gl.GLScatterPlotItem(pos=cone.pts - cone.pts.mean(0), size=1, color=(1,0,0,1))
pqt_window.addItem(cone_scatter)

# m = pts.mean(0)
m = 0
a1, a2 = m - 10*d_vector, m - 10*d_vector2
b1, b2 = m + 10*d_vector, m + 10*d_vector2

l_graph = gl.GLLinePlotItem(pos=np.array([a1, b1]), color=[1,1,1,1])
pqt_window.addItem(l_graph)

l_graph2 = gl.GLLinePlotItem(pos=np.array([a2, b2]), color=[0,0,1,1])
pqt_window.addItem(l_graph2)
