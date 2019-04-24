# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from distutils.core import setup, Extension
from distutils.command.build import build
import os
import sys

# Replicating the methods used in the RAVEN Makefile to find CROW_DIR,
# If the Makefile changes to be more robust, so should this
# We should be doing a search for CROW, I would think, we should not force a
# directory structure
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CROW_DIR = os.path.join(CURR_DIR,'crow')

BOOST_INCLUDE_DIR = os.path.join(CROW_DIR,'contrib','include')
RAVEN_INCLUDE_DIR = os.path.join('include','contrib')

# We need a custom build order in order to ensure that amsc.py is available
# before we try to copy it to the target location
class CustomBuild(build):
    sub_commands = [('build_ext', build.has_ext_modules),
                    ('build_py', build.has_pure_modules),
                    ('build_clib', build.has_c_libraries),
                    ('build_scripts', build.has_scripts)]

include_dirs=[RAVEN_INCLUDE_DIR,BOOST_INCLUDE_DIR]
if sys.version_info.major > 2:
  swig_opts=['-c++','-I'+RAVEN_INCLUDE_DIR, '-I'+BOOST_INCLUDE_DIR,'-py3']
else:
  swig_opts=['-c++','-I'+RAVEN_INCLUDE_DIR, '-I'+BOOST_INCLUDE_DIR]
extra_compile_args=['-std=c++11']
setup(name='amsc',
      version='0.0',
      description='A library for computing the Approximate Morse-Smale Complex (AMSC)',
      ext_modules=[Extension('_amsc',['src/contrib/amsc.i',
                                      'src/contrib/UnionFind.cpp',
                                      'src/contrib/AMSC.cpp'],
                             include_dirs=include_dirs, swig_opts=swig_opts,extra_compile_args=extra_compile_args)],
      package_dir={'':'src/contrib'},
      py_modules=['amsc'],
      cmdclass={'build': CustomBuild})
