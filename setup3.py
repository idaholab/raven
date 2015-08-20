from distutils.core import setup, Extension
from distutils.command.build import build

# We need a custom build order in order to ensure that amsc.py is available
# before we try to copy it to the target location
class CustomBuild(build):
    sub_commands = [('build_ext', build.has_ext_modules),
                    ('build_py', build.has_pure_modules),
                    ('build_clib', build.has_c_libraries),
                    ('build_scripts', build.has_scripts)]

include_dirs=['include/contrib']
swig_opts=['-c++','-Iinclude/contrib','-py3']
setup(name='amsc',
      version='0.0',
      description='A library for computing the Approximate Morse-Smale Complex (AMSC)',
      ext_modules=[Extension('_amsc',['src/contrib/amsc.i',
                                      'src/contrib/UnionFind.cpp',
                                      'src/contrib/AMSC.cpp'],
                             include_dirs=include_dirs, swig_opts=swig_opts)],
      package_dir={'':'src/contrib'},
      py_modules=['amsc'],
      cmdclass={'build': CustomBuild})