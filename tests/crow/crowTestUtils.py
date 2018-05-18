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
import numpy as np
import os
import importlib

ravenDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("ravenDir",ravenDir)
sys.path.append(os.path.join(ravenDir,"crow","install"))
sys.path.append(os.path.join(ravenDir,"framework"))
from utils import utils
findCrowModule = utils.findCrowModule

def checkAnswer(comment,value,expected,results,tol=1e-10):
  """ Will check if a test passes or fails and update the results dictionary.
    @ In, comment: A user-specified comment that will be printed with the test
                   case.
    @ In, value: the generated test value.
    @ In, expected: the gold standard to which value will be compared.
    @ InOut, results: a dictionary to which pass or fail will be incremented.
    @ In, tol: an optional tolerance value specifying how close expected and
               value should be.
  """
  if abs(value - expected) > tol:
    print("checking answer",comment,value,"!=",expected)
    results["fail"] += 1
  else:
    results["pass"] += 1

def checkArrayAllClose(comment,value,expected,results,tol=1e-11):
  """ Will check if a test passes or fails and update the results dictionary.
    @ In, comment: A user-specified comment that will be printed with the test
                   case.
    @ In, value: the generated test array.
    @ In, expected: the gold standard to which array will be compared.
    @ InOut, results: a dictionary to which pass or fail will be incremented.
    @ In, tol: an optional tolerance value specifying how close expected and
               value should be.
  """
  if np.allclose(value,expected,rtol=tol,atol=tol):
    results["pass"] += 1
  else:
    print("checking answer",comment,value,"!=",expected)
    results["fail"] += 1
