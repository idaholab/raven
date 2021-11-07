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
  This Module performs Unit Tests for the cached_ndarray module
  It cannot be considered part of the active code but of the regression test system
"""

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os,sys
import numpy as np

# numpy with version 1.14.0 and upper will change the floating point type and print
# https://docs.scipy.org/doc/numpy-1.14.0/release.html
if int(np.__version__.split('.')[1]) > 13:
  np.set_printoptions(**{'legacy':'1.13'})

frameworkDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),os.pardir,os.pardir,os.pardir,os.pardir,'framework'))
sys.path.append(frameworkDir)
from utils import cached_ndarray
print (cached_ndarray)


results = {"pass":0,"fail":0}


def checkAnswer(comment,value,expected,tol=1e-10,updateResults=True):
  """
    This method is aimed to compare two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ In, updateResults, bool, optional, if True updates global results
    @ Out, None
  """
  if abs(value - expected) > tol:
    print("checking answer",comment,value,"!=",expected)
    if updateResults:
      results["fail"] += 1
    return False
  else:
    if updateResults:
      results["pass"] += 1
    return True


#establish test array
origin = np.array([-3.14,2.99792,2.718,8.987,0.618])
#test init
testArray = cached_ndarray.c1darray(values=origin)

#test iter, getitem
for i,val in enumerate(testArray):
  checkAnswer('content storage indexing',val,origin[i])

#test len
checkAnswer('array length',len(testArray),5)

#test append single value
testArray.append(-6.626)
checkAnswer('append value',testArray[-1],-6.626)
#test append array
testArray.append(np.array([12.56,6.67]))
checkAnswer('append array, 0',testArray[-2],12.56)
checkAnswer('append array, 1',testArray[-1],6.67)

#test return closest
right = [4,6,5]
for f,find in enumerate([0.6,1e10,-1e10]):
  closest = testArray.returnIndexClosest(find)
  checkAnswer('find closest %1.1e' %find,closest,right[f])

#test returnIndexFirstPassage
checkAnswer('index first passage',testArray.returnIndexFirstPassage(3),3)

#test max
checkAnswer('index max',testArray.returnIndexMax(),6)

#test min
checkAnswer('index min',testArray.returnIndexMin(),5)


#test repr
msg = str(testArray)
right = 'array([ -3.14   ,   2.99792,   2.718  ,   8.987  ,   0.618  ,  -6.626  ,\n        12.56   ,   6.67   ])'

if msg == right:
  results['pass']+=1
else:
  print('checking string representation does not match:\n'+msg,'\n!=\n'+right)
  results['fail']+=1

print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.cachedNDArray</name>
    <author>talbpaul</author>
    <created>2016-11-01</created>
    <classesTested>utils.cachedNDArray</classesTested>
    <description>
       This test performs Unit Tests for the cached_ndarray module
       It cannot be considered part of the active code but of the regression test system
    </description>
    <revisions>
      <revision author="talbpaul" date="2016-11-08">Relocated utils tests</revision>
      <revision author="alfoa" date="2017-01-21">Adding this test description.</revision>
    </revisions>
  </TestInfo>
"""
