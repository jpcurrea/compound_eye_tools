from distutils.core import setup


setup(
    name='compound_eye_tools',
    version='0.1.0',
    author='John Paul Currea',
    author_email='johnpaulcurrea@gmail.com',
    packages=['compound_eye_tools'],
    scripts=[
        './bin/0a_filter_data.py', './bin/0b_eye_cluster.py', './bin/1_4_get_cone_clusters.py',
        './bin/5_take_measurements.py', './bin/extract_cone_centers.py'],
    url='https://github.com/jpcurrea/compound_eye_tools',
    license='LICENSE.txt',
    description='This package offers tools and a general pipeline/interface ' +
    'for analyzing microCT and microscope image stacks for relevant optical '+
    'measurements (inter-ommatidial angle, ommatidial area, ommatia count, etc.).',
    long_description=open('README.txt').read(),
    install_requires=[
        'numpy'
        'matplotlib',
        'scipy',
        'pandas',
        'PIL',
        'sty',
        'hdbscan'
    ],
)
