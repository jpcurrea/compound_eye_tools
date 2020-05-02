# from distutils.core import setup
from setuptools import setup


setup(
    name='compound_eye_tools',
    version='0.1.0',
    author='John Paul Currea',
    author_email='johnpaulcurrea@gmail.com',
    packages=['compound_eye_tools'],
    # scripts=['./bin/extract_cones.py'],
    # entry_points={
    #     'console_scripts':['extract-cones=compound_eye_tools.command_line:main'],
    #     },
    url='https://github.com/jpcurrea/compound_eye_tools',
    license='LICENSE.txt',
    description='This package offers tools and a general pipeline/interface ' +
    'for analyzing microCT and microscope image stacks for relevant optical ' +
    'measurements (inter-ommatidial angle, ommatidial area, ommatia count, etc.).',
    long_description=open('README.txt').read(),
    install_requires=[
        'fly_eye',
        'matplotlib',
        'numpy',
        'pandas',
        'pillow',
        'pyqt5',
        'PyOpenGL',
        'pyqtgraph',
        'scipy',
        'scikit-learn',
        'seaborn',
    ],
    dependency_links=[
        'git+https://github.com/jpcurrea/fly_eye#egg=fly_eye-0.1',
    ],
    scripts=['./bin/process-eye']
)
