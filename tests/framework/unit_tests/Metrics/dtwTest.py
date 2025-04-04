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
  Testing for the onePointCrossover method
  @authors: Mohammad Abdo, Diego Mandelli, Andrea Alfonsi
"""
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET

ravenDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4)))
sys.path.append(ravenDir)
frameworkDir = os.path.join(ravenDir, 'framework')

from ravenframework.Metrics.metrics import DTW
from ravenframework import MessageHandler

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug'})

#
#
# checkers
#

results = {"pass":0,"fail":0}

def checkAnswer(comment,value,expected,tol=1e-10,relative=False):
  """
    This method is aimed to compare two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float (or string), the value to compare
    @ In, expected, float (or string), the expected value
    @ In, tol, float, optional, the tolerance (valid for floats/ints only)
    @ In, relative, bool, optional, the tolerance needs be checked relative?
    @ Out, None
  """
  isFloat = True
  try:
    val, expect = float(value), float(expected)
  except ValueError:
    val, expect = value, expected
    isFloat = False
  if relative and isFloat:
    denominator = expect if expect != 0. else 1.0
  if isFloat:
    diff = abs(val - expect) if not relative else abs(val - expect)/denominator
  else:
    diff = 0.0 if val == expect else tol + 1.0

  if diff > tol:
    print("checking answer",comment,val,"!=",expect)
    results["fail"] += 1
  else:
    results["pass"] += 1

#
#
# initialize metric instance
#    <Metric name="dtw" subType="DTW">
#      <order>0</order>
#      <localDistance>euclidean</localDistance>
#    </Metric>
#

DTWinstance = DTW.DTW()
DTWinstance.order=0
DTWinstance.localDistance = "euclidean"

#
#
# initialize test
#
s1 = np.array([1, 2, 3, 2, 2.13, 1])
s2 = np.array([1, 1, 2, 2, 2.42, 3, 2, 1])

DTWdistance = DTWinstance.run(s1,s2)

## TESTING
# Test DTW distance metric
expectedDTWdistance = 0.5499999999999998
checkAnswer('DTW analytical test', DTWdistance, expectedDTWdistance)
#
# end
#
print('Results:', results)
sys.exit(results['fail'])
