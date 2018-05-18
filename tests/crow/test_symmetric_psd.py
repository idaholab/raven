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
# This tests is used to verify the nearest symmetric positive semidefinite matrix calculated inside crow
# the eig module from numpy.linalg is used as gold solution.

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
cov = [0.21,   -0.16,  0.21,  0.43,    -0.144,
       -0.17, 0.59,  0.794,  -0.173,  -0.406,
       0.34,  0.794,   5.41, 0.461,   0.179,
       0.43,   -0.13,  0.41,  0.3,   0.822,
       -0.144, -0.46,  0.19,  0.82,   3.75]
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

dim = mvnDistribution.getSingularValuesDimension()
sCpp_vector = mvnDistribution.getSingularValues()
sCpp = [sCpp_vector[i] for i in range(dim)]
sCpp = np.asarray(sCpp)
dimVectorLeft = mvnDistribution.getLeftSingularVectorsDimensions()
uCpp_vector = mvnDistribution.getLeftSingularVectors()
uCpp = [uCpp_vector[i] for i in range(dimVectorLeft[0]*dimVectorLeft[1])]
uCpp = np.asarray(uCpp)
uCpp = np.reshape(uCpp,(dimVectorLeft[0],dimVectorLeft[1]))

# using numpy to compute the eigenvalues
cov = np.asarray(cov)
# compute the symmetric matrix
covNp = cov.reshape(5,5)
covNp = (covNp + covNp.T)/2.0
sNp,uNp = LA.eig(covNp)
#reset the singular values
for i in range(sNp.size):
  if sNp[i] < 0: sNp[i] = 0.0
#reorder the singular values and left singular vectors
indexNp = np.argsort(sNp)
indexCpp = np.argsort(sCpp)
sNp = sNp[indexNp]
sCpp = sCpp[indexCpp]
uCpp = uCpp[:,indexCpp]
uNp = uNp[:,indexNp]

results = {"pass":0,"fail":0}

utils.checkArrayAllClose("Singular values",sCpp,sNp,results)
utils.checkArrayAllClose("Left singular vectors",np.absolute(uCpp),np.absolute(uNp),results)

print(results)

sys.exit(results["fail"])

"""
 <TestInfo>
    <name>crow.test_inverse_transformationMatrix</name>
    <author>cogljj</author>
    <created>2017-03-24</created>
    <classesTested>crow</classesTested>
    <description>
      This test is a Unit Test for the crow swig classes. It tests that the MultiVariate Normal
      distribution is accessable by Python and that the lef singular vectors are correctly
      reported
    </description>
    <revisions>
      <revision author="alfoa" date="2018-05-15">Adding this test description.</revision>
    </revisions>
 </TestInfo>
"""
