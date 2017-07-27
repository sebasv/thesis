"""
python setup.py build_ext -i
"""

import shutil
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Hello world app',
    ext_modules=cythonize(r"D:\git\thesis\optimization\Optimizer\coptimizer\coptimizer.pyx"),
    include_dirs=[numpy.get_include()]
)

shutil.move(
    r'coptimizer.cp36-win_amd64.pyd', 
    r'd:\git\thesis\optimization\Optimizer\coptimizer.pyd'
)
