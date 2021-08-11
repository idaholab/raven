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
"""
Created on 2017-September-8

@author: cogljj

This is used to test the comparison statistics
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import sys, os

ravenDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
print(ravenDir)
frameworkDir = os.path.join(ravenDir,"framework")
sys.path.append(frameworkDir)

import utils.utils

utils.utils.find_crow(frameworkDir)

from Metrics.metrics import MetricUtilities
import Distributions

print(dir(MetricUtilities))

count = MetricUtilities._countWeightInBins([(1.0,0.5),(2.0,0.5)],[1.5])

print(count)

assert count == [0.5, 0.5]

simple = range(64)
simple_prob = [1.0/64.0]*64

stats, cdf, pdf = MetricUtilities._getPDFandCDFfromWeightedData(simple,simple_prob,3,False,"linear")
print(stats)
print(cdf(0.0),cdf(32.0),cdf(64.0))
assert 0.3 < cdf(32.0) < 0.7

stats, cdf, pdf = MetricUtilities._getPDFandCDFfromWeightedData(simple,simple_prob,3,True,"linear")
print(stats)
print(cdf(0.0),cdf(32.0),cdf(64.0))
assert 0.3 < cdf(32.0) < 0.7


stats, cdf, pdf = MetricUtilities._getPDFandCDFfromWeightedData(simple,simple_prob,12,False,"linear")
print(stats)
print(cdf(0.0),cdf(32.0),cdf(64.0))
assert 0.4 < cdf(32.0) < 0.6


stats, cdf, pdf = MetricUtilities._getPDFandCDFfromWeightedData(simple,simple_prob,12,True,"linear")
print(stats)
print(cdf(0.0),cdf(32.0),cdf(64.0))
assert 0.4 < cdf(32.0) < 0.6

low, high = MetricUtilities._getBounds({"low":1.0,"high":3.0},{"low":2.0,"high":2.5})
assert low == 1.0
assert high == 3.0

dist1 = Distributions.Normal(0.0, 1.0)
dist1.initializeDistribution()

dist2 = Distributions.Normal(1.0, 1.0)
dist2.initializeDistribution()

#Test same
cdfAreaDifference = MetricUtilities._getCDFAreaDifference(dist1, dist1)

print("cdfAreaDifference same",cdfAreaDifference)
assert -1e-3 < cdfAreaDifference < 1e-3

pdfCommonArea = MetricUtilities._getPDFCommonArea(dist1, dist1)

print("pdfCommonArea same",pdfCommonArea)
assert 0.99 < pdfCommonArea < 1.01

#Test different
cdfAreaDifference = MetricUtilities._getCDFAreaDifference(dist1, dist2)

print("cdfAreaDifference different",cdfAreaDifference)
assert 0.99 < cdfAreaDifference < 1.01

pdfCommonArea = MetricUtilities._getPDFCommonArea(dist1, dist2)

print("pdfCommonArea different",pdfCommonArea)
assert 0.60 < pdfCommonArea < 0.62

"""
  <TestInfo>
    <name>framework.test_distributions</name>
    <author>cogljj</author>
    <created>2017-09-08</created>
    <classesTested> </classesTested>
    <description>
       This test is a Unit Test for the comparison statistics metric. It checks all the functions that are available to RAVEN
       internally.
    </description>
    <revisions>
      <revision author="alfoa" date="2018-05-10">Added Log Uniform distribution unit test</revision>
    </revisions>
  </TestInfo>
"""



