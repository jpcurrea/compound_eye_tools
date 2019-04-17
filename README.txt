===========
Compound Eye Tools
===========

This package offers tools and a general pipeline/interface for analyzing
microCT and microscope image stacks of compound eyes for relevant optical
measurements (inter-ommatidial angle, ommatidial area, ommatia count, etc.). 

MicroCT Pipeline
=========

The microCT pipeline segments individual crystalline cones from a stack of images:

0. segment the eye using a density filter and HDBSCAN
1. convert to spherical coorinates by fitting a sphere with OLS
2. model each r as a function of theta and phi.
3. extract a range of points around the approximate surface
4. convert cone centers back to cartesian coordinates. For each center, find the nearest cluster of points within a generous radius.
5. Using our set of cone clusters, we can take measurements relevant to the eye's optics.
