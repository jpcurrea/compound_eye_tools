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
        clusterer = cluster.KMeans(n_clusters=len(
            cluster_centers), init=cluster_centers).fit(eye.pts)
        groups = clusterer.labels_
        clusters = []
        for group in sorted(set(groups)):
            ind = group == groups
            cone = Points(eye.pts[ind], polar=eye.polar[ind],
                          center_points=False, rotate_com=False)
            clusters += [cone]
    scatter_centers = ScatterPlot3d(
        cluster_centers,
        size=10,
    )
    scatters = []
    for cluster_ in clusters:
        color = tuple(np.append(np.random.random(size=3), [1]))
        scatter_cluster = ScatterPlot3d(cluster_.pts, color=color,
                                        window=scatter_centers.window)
    scatter_centers.show()
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
    neighbor_lbls = np.array([cone.neighbor_lbl for cone in cones])

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
        else:
            anatomical_angle = interommatidial_angle_anatomical[pair]
            approx_angle = interommatidial_angle_approx[pair]
        anatomical_IOAs += [anatomical_angle]
        approx_IOAs += [approx_angle]
    cone.approx_FOV = np.mean(approx_IOAs)
    cone.anatomical_FOV = np.mean(anatomical_IOAs)
    print_progress(num, len(cones))
