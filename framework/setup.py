from distutils.core import setup
from Cython.Build import cythonize

setup(name='GridEntities',zip_safe=False,
      ext_modules=cythonize("GridEntities.pyx"))
