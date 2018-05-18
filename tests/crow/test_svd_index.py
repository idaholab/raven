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
# This test is used to verify functions that used to obtain the information from the svd decomposition calculated inside crow
# the svd module from numpy.linalg is used as gold solution.

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

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

# Transform 'mu' and 'cov' to the c++ vector
muCpp = distribution1D.vectord_cxx(len(mu))
for i in range(len(mu)):
  muCpp[i] = mu[i]
covCpp = distribution1D.vectord_cxx(len(cov))
for i in range(len(cov)):
  covCpp[i] = cov[i]

# Index set used to obtian the information from SVD calculation
index = [4,2,3,1]
indexCpp = distribution1D.vectori_cxx(len(index))
for i in range(len(index)):
  indexCpp[i] = index[i]

# call the functions from the crow to compute the svd
covType = "abs"
rank = 5
mvnDistribution = distribution1D.BasicMultivariateNormal(covCpp,muCpp,str(covType),rank)

dim = mvnDistribution.getSingularValuesDimension(indexCpp)
sCpp_vector = mvnDistribution.getSingularValues(indexCpp)
sCpp = [sCpp_vector[i] for i in range(dim)]
sCpp = np.asarray(sCpp)

dimVectorLeft = mvnDistribution.getLeftSingularVectorsDimensions(indexCpp)
uCpp_vector = mvnDistribution.getLeftSingularVectors(indexCpp)
uCpp = [uCpp_vector[i] for i in range(dimVectorLeft[0]*dimVectorLeft[1])]
uCpp = np.asarray(uCpp)
uCpp = np.reshape(uCpp,(dimVectorLeft[0],dimVectorLeft[1]))

dimVectorRight = mvnDistribution.getRightSingularVectorsDimensions(indexCpp)
vCpp_vector = mvnDistribution.getRightSingularVectors(indexCpp)
vCpp = [vCpp_vector[i] for i in range(dimVectorRight[0]*dimVectorRight[1])]
vCpp = np.asarray(vCpp)
vCpp = np.reshape(vCpp,(dimVectorRight[0],dimVectorRight[1]))

# using numpy to compute the svd
COVn = np.asarray(cov).reshape(-1,int(sqrt(len(cov))))
Un,Sn,Vn = LA.svd(COVn,full_matrices=False)
uNp = Un[:,index]
sNp = Sn[index]
vNp = Vn[index,:]
covNp = np.dot(uNp,np.dot(np.diag(sNp),vNp))

# reconstruct the matrix using A = U*S*V.T
covReCompute = np.dot(uCpp,np.dot(np.diag(sCpp),vCpp.T))



results = {"pass":0,"fail":0}

utils.checkArrayAllClose("MVN singular values",sCpp,sNp,results)
utils.checkArrayAllClose("MVN left singular vectors",np.absolute(uCpp),np.absolute(uNp),results)
utils.checkArrayAllClose("MVN right singular vectors", np.absolute(vCpp),np.absolute(vNp.T),results)
utils.checkArrayAllClose("MVN singular value decomposition",covNp,covReCompute,results)

utils.checkAnswer("MVN dimensions of singular values",dim,len(index),results)
utils.checkAnswer("MVN row dimensions of left singular vectors",dimVectorLeft[0],5,results)
utils.checkAnswer("MVN col dimensions of left singular vectors",dimVectorLeft[1],len(index),results)
utils.checkAnswer("MVN row dimensions of right singular vectors",dimVectorRight[0],5,results)
utils.checkAnswer("MVN col dimensions of right singular vectors",dimVectorRight[1],len(index),results)

print(results)

sys.exit(results["fail"])

"""
 <TestInfo>
    <name>crow.test_svd_index</name>
    <author>cogljj</author>
    <created>2017-03-24</created>
    <classesTested>crow</classesTested>
    <description>
    This test is a Unit Test for the crow swig classes. It tests that the MultiVariate Normal
    distribution is accessable by Python and that the SVD indexes are accessable
    </description>
    <revisions>
      <revision author="alfoa" date="2018-05-15">Adding this test description.</revision>
    </revisions>
 </TestInfo>
"""
