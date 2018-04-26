from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='sammpler',
    ext_modules=cythonize("sampler.pyx"),
)
