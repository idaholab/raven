from distutils.core import setup, Extension
include_dirs=['include/contrib']
swig_opts=['-c++','-Iinclude/contrib','-py3']
setup(name='amsc',
      version='0.0',
      description='A library for computing the Approximate Morse-Smale Complex (AMSC)',
      ext_modules=[Extension('_amsc',['src/contrib/amsc.i',
                                      'src/contrib/DenseVector.cpp',
                                      'src/contrib/DenseMatrix.cpp',
                                      'src/contrib/UnionFind.cpp',
                                      'src/contrib/AMSC.cpp'],
                   include_dirs=include_dirs, swig_opts=swig_opts)])
