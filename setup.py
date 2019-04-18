from distutils.core import setup


setup(
    name='compound_eye_tools',
    version='0.1.0',
    author='John Paul Currea',
    author_email='johnpaulcurrea@gmail.com',
    packages=['compound_eye_tools'],
    scripts=[
        './bin/filter_data.py', './bin/eye_segmentation.py',
        './bin/crystalline_cone_segmentation.py',
        './bin/eye_measurements.py', './bin/extract_cones.py'],
    url='https://github.com/jpcurrea/compound_eye_tools',
    license='LICENSE.txt',
    description='This package offers tools and a general pipeline/interface ' +
    'for analyzing microCT and microscope image stacks for relevant optical '+
    'measurements (inter-ommatidial angle, ommatidial area, ommatia count, etc.).',
    long_description=open('README.txt').read(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'pandas',
        'pillow',
        'sty',
        'hdbscan'
    ],
    dependency_links=[
        'git+https://github.com/scikit-learn-contrib/hdbscan.git#egg=hdbscan'
        ]
)
