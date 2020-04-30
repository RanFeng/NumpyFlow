from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name='numcy',
    ext_modules=cythonize("numcy.pyx"),
    extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)

