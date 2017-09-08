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

import PostProcessors.ComparisonStatisticsModule

print(dir(PostProcessors.ComparisonStatisticsModule))

count = PostProcessors.ComparisonStatisticsModule._countWeightInBins([(1.0,0.5),(2.0,0.5)],[1.5])

print(count)

assert count == [0.5, 0.5]

simple = range(64)
simple_prob = [1.0/64.0]*64

stats, cdf, pdf = PostProcessors.ComparisonStatisticsModule._getPDFandCDFfromWeightedData(simple,simple_prob,3,False,"linear")
print(stats)
print(cdf(0.0),cdf(32.0),cdf(64.0))
assert 0.3 < cdf(32.0) < 0.7

stats, cdf, pdf = PostProcessors.ComparisonStatisticsModule._getPDFandCDFfromWeightedData(simple,simple_prob,3,True,"linear")
print(stats)
print(cdf(0.0),cdf(32.0),cdf(64.0))
assert 0.3 < cdf(32.0) < 0.7


stats, cdf, pdf = PostProcessors.ComparisonStatisticsModule._getPDFandCDFfromWeightedData(simple,simple_prob,12,False,"linear")
print(stats)
print(cdf(0.0),cdf(32.0),cdf(64.0))
assert 0.4 < cdf(32.0) < 0.6


stats, cdf, pdf = PostProcessors.ComparisonStatisticsModule._getPDFandCDFfromWeightedData(simple,simple_prob,12,True,"linear")
print(stats)
print(cdf(0.0),cdf(32.0),cdf(64.0))
assert 0.4 < cdf(32.0) < 0.6


