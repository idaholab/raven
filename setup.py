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
import setuptools

# Replicating the methods used in the RAVEN Makefile to find CROW_DIR,
# If the Makefile changes to be more robust, so should this
# We should be doing a search for CROW, I would think, we should not force a
# directory structure
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CROW_DIR = os.path.join(CURR_DIR,'crow')

BOOST_INCLUDE_DIR = os.path.join(CROW_DIR,'contrib','include')
RAVEN_INCLUDE_DIR = os.path.join('include','contrib')
DIST_INCLUDE_DIR = os.path.join(CROW_DIR,'include', 'distributions')
UTIL_INCLUDE_DIR = os.path.join(CROW_DIR,'include', 'utilities')

# We need a custom build order in order to ensure that amsc.py is available
# before we try to copy it to the target location
class CustomBuild(build):
    sub_commands = [('build_ext', build.has_ext_modules),
                    ('build_py', build.has_pure_modules),
                    ('build_clib', build.has_c_libraries),
                    ('build_scripts', build.has_scripts)]


include_dirs=[RAVEN_INCLUDE_DIR,BOOST_INCLUDE_DIR, DIST_INCLUDE_DIR, UTIL_INCLUDE_DIR]
swig_opts=['-c++','-I'+RAVEN_INCLUDE_DIR, '-I'+BOOST_INCLUDE_DIR,'-I'+DIST_INCLUDE_DIR, '-I'+UTIL_INCLUDE_DIR, '-py3']
extra_compile_args=['-std=c++11']
try:
    eigen_flags = subprocess.check_output(["./scripts/find_eigen.py"]).decode("ascii")
except:
  eigen_flags = ""
if eigen_flags.startswith("-I"):
  include_dirs.append(eigen_flags[2:].rstrip())
long_description = open("README.md", "r").read()
setup(name='raven_framework',
      version='2.3',
      description='RAVEN (Risk Analysis Virtual Environment) is designed to perform parametric and probabilistic analysis based on the response of complex system codes. RAVEN C++ dependenciences including a library for computing the Approximate Morse-Smale Complex (AMSC) and Crow probability tools',
      long_description=long_description,
      url="https://raven.inl.gov/",
      package_dir={'AMSC': 'src/AMSC', 'crow_modules': 'src/crow_modules', 'ravenframework': 'ravenframework'},
      classifiers=['Programming Language :: Python :: 3'],
      entry_points={
          'console_scripts': [
              'raven_framework = ravenframework.Driver:wheelMain'
          ]
      },
      ext_modules=[
          Extension('crow_modules._distribution1D',
                  ['src/crow_modules/distribution1D.i',
                   'src/distributions/distribution.cxx',
                   'src/utilities/MDreader.cxx',
                   'src/utilities/inverseDistanceWeigthing.cxx',
                   'src/utilities/microSphere.cxx',
                   'src/utilities/NDspline.cxx',
                   'src/utilities/ND_Interpolation_Functions.cxx',
                   'src/distributions/distributionNDBase.cxx',
                   'src/distributions/distributionNDNormal.cxx',
                   'src/distributions/distributionFunctions.cxx',
                   'src/distributions/DistributionContainer.cxx',
                   'src/distributions/distribution_1D.cxx',
                   'src/distributions/randomClass.cxx',
                   'src/distributions/distributionNDCartesianSpline.cxx'],
                include_dirs=include_dirs,
                swig_opts=swig_opts,
                extra_compile_args=extra_compile_args),
          Extension('crow_modules._randomENG',['src/crow_modules/randomENG.i','src/distributions/randomClass.cxx'],include_dirs=include_dirs,swig_opts=swig_opts,extra_compile_args=extra_compile_args),
          Extension('crow_modules._interpolationND',['src/crow_modules/interpolationND.i','src/utilities/ND_Interpolation_Functions.cxx','src/utilities/NDspline.cxx','src/utilities/microSphere.cxx','src/utilities/inverseDistanceWeigthing.cxx','src/utilities/MDreader.cxx','src/distributions/randomClass.cxx'],include_dirs=include_dirs,swig_opts=swig_opts,extra_compile_args=extra_compile_args),
          Extension('AMSC._amsc',['src/AMSC/amsc.i',
                             'src/AMSC/UnionFind.cpp',
                             'src/AMSC/AMSC.cpp'],
                    include_dirs=include_dirs, swig_opts=swig_opts,extra_compile_args=extra_compile_args)],
      py_modules=['AMSC.amsc','crow_modules.distribution1D','crow_modules.randomENG','crow_modules.interpolationND', 'AMSC.AMSC_Object'],
      packages=['ravenframework.'+x for x in setuptools.find_packages('ravenframework')]+['ravenframework'],
      cmdclass={'build': CustomBuild})
