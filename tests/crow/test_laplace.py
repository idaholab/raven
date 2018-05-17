#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys
import crowTestUtils as utils

distribution1D = utils.findCrowModule('distribution1D')

laplace_distribution = distribution1D.BasicLaplaceDistribution(0.0, 2.0, -sys.float_info.max, sys.float_info.max)

results = {"pass":0,"fail":0}

utils.checkAnswer("normal cdf(0.0)",laplace_distribution.cdf(0.0),0.5,results)
utils.checkAnswer("normal cdf(1.0)",laplace_distribution.cdf(1.0),0.696734670144,results)
utils.checkAnswer("normal cdf(2.0)",laplace_distribution.cdf(2.0),0.816060279414,results)

utils.checkAnswer("normal mean",laplace_distribution.untrMean(),0.0,results)
utils.checkAnswer("normal stddev",laplace_distribution.untrStdDev(),2.82842712475,results)
utils.checkAnswer("normal ppf(0.1)",laplace_distribution.inverseCdf(0.1),-3.21887582487,results)
utils.checkAnswer("normal ppf(0.5)",laplace_distribution.inverseCdf(0.5),0.0,results)
utils.checkAnswer("normal ppf(0.9)",laplace_distribution.inverseCdf(0.9),3.21887582487,results)
utils.checkAnswer("normal mean()",laplace_distribution.untrMean(),0.0,results)
utils.checkAnswer("normal median()",laplace_distribution.untrMedian(),0.0,results)
utils.checkAnswer("normal mode()",laplace_distribution.untrMode(),0.0,results)


print(results)

sys.exit(results["fail"])

"""
 <TestInfo>
    <name>crow.test_laplace</name>
    <author>cogljj</author>
    <created>2017-03-24</created>
    <classesTested>crow</classesTested>
    <description>
      This test is a Unit Test for the crow swig classes. It tests that the laplace
      distribution is accessable by Python
    </description>
    <revisions>
      <revision author="alfoa" date="2018-05-15">Adding this test description.</revision>
    </revisions>
 </TestInfo>
"""
