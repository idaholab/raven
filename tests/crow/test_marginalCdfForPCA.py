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
#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys
import crowTestUtils as utils

distribution1D = utils.findCrowModule('distribution1D')
# input data, random matrix can also be used.
mu = [1.0,2.0,3.0,4.0,5.0]
cov = [1.36,   -0.816,  0.521,  1.43,    -0.144,
       -0.816, -0.659,  0.794,  -0.173,  -0.406,
       0.521,  0.794,   -0.541, 0.461,   0.179,
       1.43,   -0.173,  0.461,  -1.43,   0.822,
       -0.144, -0.406,  0.179,  0.822,   -1.37]

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
mvnDistribution = distribution1D.BasicMultivariateNormal(covCpp,muCpp,str(covType),rank)

results = {"pass":0,"fail":0}
# the marginal cdf is defined as standard norm distribution with mean 0 and variance 1
utils.checkAnswer("multivariate norm marginal for pca (-4.0)",mvnDistribution.marginalCdfForPCA(-4.0),3.1671241833119863e-05,results)
utils.checkAnswer("multivariate norm marginal for pca (-3.5)",mvnDistribution.marginalCdfForPCA(-3.5),0.00023262907903552502,results)
utils.checkAnswer("multivariate norm marginal for pca (-2.2)",mvnDistribution.marginalCdfForPCA(-2.2),0.013903447513498595,results)
utils.checkAnswer("multivariate norm marginal for pca (-1.1)",mvnDistribution.marginalCdfForPCA(-1.1),0.13566606094638267,results)
utils.checkAnswer("multivariate norm marginal for pca (-0.7)",mvnDistribution.marginalCdfForPCA(-0.7),0.24196365222307303,results)
utils.checkAnswer("multivariate norm marginal for pca (4.0)",mvnDistribution.marginalCdfForPCA(4.0),0.99996832875816688,results)
utils.checkAnswer("multivariate norm marginal for pca (3.5)",mvnDistribution.marginalCdfForPCA(3.5),0.99976737092096446,results)
utils.checkAnswer("multivariate norm marginal for pca (2.2)",mvnDistribution.marginalCdfForPCA(2.2),0.98609655248650141,results)
utils.checkAnswer("multivariate norm marginal for pca (1.1)",mvnDistribution.marginalCdfForPCA(1.1),0.86433393905361733,results)
utils.checkAnswer("multivariate norm marginal for pca (0.7)",mvnDistribution.marginalCdfForPCA(0.7),0.75803634777692697,results)
utils.checkAnswer("multivariate norm marginal for pca (0.0)",mvnDistribution.marginalCdfForPCA(0.0),0.5,results)

print(results)

sys.exit(results["fail"])

"""
 <TestInfo>
    <name>crow.test_marginalCdfForPCA</name>
    <author>cogljj</author>
    <created>2017-03-24</created>
    <classesTested>crow</classesTested>
    <description>
      This test is a Unit Test for the crow swig classes. It tests that the MultiVariate Normal
      distribution is accessable by Python and that PCA is correctly performed and that the
      CDF marginal distribution is computable in the transformed space
    </description>
    <revisions>
      <revision author="alfoa" date="2018-05-15">Adding this test description.</revision>
    </revisions>
 </TestInfo>
"""
