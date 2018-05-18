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
# This tests is used to verify the svd decomposition calculated inside crow
# the svd module from numpy.linalg is used as gold solution.

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#!/usr/bin/env python

import sys
import crowTestUtils as utils
import numpy as np
from math import sqrt
from numpy import linalg as LA

distribution1D = utils.findCrowModule('distribution1D')
# input data, random matrix can also be used.
mu = [1.0,2.0,3.0,4.0,5.0]
cov = [1.36,   -0.16,  0.21,  0.43,    -0.144,
       -0.16, 6.59,  0.794,  -0.173,  -0.406,
       0.21,  0.794,   5.41, 0.461,   0.179,
       0.43,   -0.173,  0.461,  14.3,   0.822,
       -0.144, -0.406,  0.179,  0.822,   3.75]

#dim = 5

# Transform 'mu' and 'cov' to the c++ vector
muCpp = distribution1D.vectord_cxx(len(mu))
for i in range(len(mu)):
  muCpp[i] = mu[i]
covCpp = distribution1D.vectord_cxx(len(cov))
for i in range(len(cov)):
  covCpp[i] = cov[i]

# call the functions from the crow to compute the svd
covType = "abs"
rank = 5
mvnDistribution = distribution1D.BasicMultivariateNormal(covCpp,muCpp,str(covType),5)

dimVector = mvnDistribution.getTransformationMatrixDimensions()
uCpp_vector = mvnDistribution.getTransformationMatrix()
uCpp = [uCpp_vector[i] for i in range(dimVector[0]*dimVector[1])]
uCpp = np.asarray(uCpp)
uCpp = np.reshape(uCpp,(dimVector[0],dimVector[1]))

# using numpy to compute the svd
covNp = np.asarray(cov).reshape(-1,int(sqrt(len(cov))))
uNp,sNp,vNp = LA.svd(covNp,full_matrices=False)

# compute the transformation matrix  = U*sqrt(S)
uReCompute = np.dot(uNp,np.sqrt(np.diag(sNp)))

results = {"pass":0,"fail":0}

utils.checkArrayAllClose("MVN transformation matrix",np.absolute(uCpp),np.absolute(uReCompute),results)
utils.checkAnswer("MVN row dimensions of transformation matrix",dimVector[0],5,results)
utils.checkAnswer("MVN col dimensions of transformation matrix",dimVector[1],5,results)

print(results)

sys.exit(results["fail"])


"""
 <TestInfo>
    <name>crow.test_transformationMatrix</name>
    <author>cogljj</author>
    <created>2017-03-24</created>
    <classesTested>crow</classesTested>
    <description>
      This test is a Unit Test for the crow swig classes. It tests that the MultiVariate Normal
      distribution is accessable by Python and that transformation matrix is accessable and
      correctly computed.
    </description>
    <revisions>
      <revision author="alfoa" date="2018-05-15">Adding this test description.</revision>
    </revisions>
 </TestInfo>
"""




