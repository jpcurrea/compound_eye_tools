import subprocess
import os

# 0a. let user select, from images at different orientations,
# the range of densities corresponding to the crystalline cones
fn = "./0a_filter_data.py"
cmd = ["python", fn]
res = subprocess.Popen(cmd)
res.wait()

# 0b. segment the eye using a density filter and HDBSCAN
fn = "./0b_eye_cluster.py"
cmd = ["python", fn]
res = subprocess.Popen(cmd)
res.wait()

# 1. convert to spherical coorinates by fitting a sphere with OLS
# 2. fit an n-degree polymonial, modelling each r as a function of theta and phi.
# 3. extract a range of points around the approximate surface
# 4. convert cone centers back to cartesian coordinates. For each center, find the nearest cluster of points within a generous radius .
fn = "./1_4_get_cone_clusters.py"
cmd = ["python", fn]
res = subprocess.Popen(cmd)
res.wait()

# 5. Using our set of cone clusters, and the curvature of the thin sheet, we can take measurements relevant to the eye's optics.
fn = "./5_take_measurements.py"
cmd = ["python", fn]
res = subprocess.Popen(cmd)
res.wait()
