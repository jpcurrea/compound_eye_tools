===========
Compound Eye Tools
===========

This package offers tools and a general pipeline/interface for analyzing
microCT and microscope image stacks of compound eyes for relevant optical
measurements (inter-ommatidial angle, ommatidial area, ommatia count, etc.). 

MicroCT Pipeline
=========

The microCT pipeline segments individual crystalline cones from a stack of images:

0. import files and filter images based on contrast values
1. convert to spherical coorinates by fitting a sphere with OLS
2. get approximate cross section of the points in spherical coordinates
3. Use HDBSCAN or low pass filter to find centers of cone clusters.
4. For each center, find the nearest cluster of points within a generous radius in cartesian coordinates.
5. Using our set of cone clusters, we can take measurements relevant to the eye's optics.
