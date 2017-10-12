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
import xarray as xr

frameworkDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),os.pardir,os.pardir,os.pardir,'framework'))
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
    print("checking answer",comment,'|',value,"!=",expected)
    if updateResults:
      results["fail"] += 1
    return False
  else:
    if updateResults:
      results["pass"] += 1
    return True


##################
# 1D Array Tests #
##################

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

##################
# ND Array Tests #
##################

## POINT SET ##

# default construction
testArray = cached_ndarray.cNDarray(width=3,length=10)
checkAnswer('initial capacity',testArray.capacity,10)
checkAnswer('initial width',testArray.shape[1],3)
checkAnswer('initial size',testArray.size,0)
checkAnswer('initial len',len(testArray),0)

#get empty
checkAnswer('getData empty size',testArray.getData().size,0)

#append entry
vals = np.array([[1.0,2.0,3.0]])
testArray.append(vals)
#check values
aValues = testArray.getData()
for v,val in enumerate(vals):
  checkAnswer('appended[{}]'.format(v),aValues[0,v],vals[0,v])

#iter
for aValues in testArray:
  for v,val in enumerate(vals):
    checkAnswer('iter[{}]'.format(v),aValues[v],vals[0,v])

# append more
vals = [0,0]
vals[0] = [11.0,12.0,13.0]
testArray.append(np.array([vals[0]]))
vals[1] = [21.0,22.0,23.0]
testArray.append(np.array([vals[1]]))
#test slicing
for a,ar in enumerate(testArray[1:]):
  for i in range(3):
    checkAnswer('slicing [{},{}]'.format(a,i),ar[i],vals[a][i])


# construction via values
values = np.array(
             [[ 1.0,  2.0,  3.0],
              [11.0, 12.0, 13.0],
              [21.0, 22.0, 23.0]]
             )
testArray = cached_ndarray.cNDarray(values=values)
for i in range(values.shape[0]):
  for j in range(values.shape[1]):
    checkAnswer('initialize by value: [{},{}]'.format(i,j),values[i][j],testArray.values[i][j])


## ND SET ##

#default construction
testArray = cached_ndarray.cNDarray(width=3,length=10,dtype=object)
checkAnswer('initial capacity',testArray.capacity,10)
checkAnswer('initial width',testArray.shape[1],3)
checkAnswer('initial size',testArray.size,0)
checkAnswer('initial len',len(testArray),0)

#append entry
vals = np.array([[1.0,
                 xr.DataArray([ 2.0, 2.1, 2.2],dims=['time'],coords={'time':[1e-6,2e-6,3e-6]}),
                 xr.DataArray([[ 3.00, 3.01, 3.02],[ 3.10, 3.11, 3.12]],dims=['space','time'],coords={'space':[1e-3,2e-3],'time':[1e-6,2e-6,3e-6]})
                 ]],dtype=object)
testArray.append(vals)
checkAnswer('ND append, point',testArray.values[0,0],1.0)
checkAnswer('ND append, hist, time 0',testArray.values[0,1][0],2.0)
checkAnswer('ND append, nd, time 0, location 0',testArray.values[0,2][0,0], 3.00)

#values construction
values = np.ndarray([3,3],dtype=object)
values[0,0] = 1.0
values[1,0] =11.0
values[2,0] =21.0

values[0,1] = xr.DataArray([ 2.0, 2.1, 2.2],dims=['time'],coords={'time':[1e-6,2e-6,3e-6]})
values[1,1] = xr.DataArray([12.0,12.1,12.2],dims=['time'],coords={'time':[1e-6,2e-6,3e-6]})
values[2,1] = xr.DataArray([22.0,22.1,22.2],dims=['time'],coords={'time':[1e-6,2e-6,3e-6]})

values[0,2] = xr.DataArray([[ 3.00, 3.01, 3.02],[ 3.10, 3.11, 3.12]],dims=['space','time'],coords={'space':[1e-3,2e-3],'time':[1e-6,2e-6,3e-6]})
values[1,2] = xr.DataArray([[13.00,13.01,13.02],[13.10,13.11,13.12]],dims=['space','time'],coords={'space':[1e-3,2e-3],'time':[1e-6,2e-6,3e-6]})
values[2,2] = xr.DataArray([[23.00,23.01,23.02],[23.10,23.11,23.12]],dims=['space','time'],coords={'space':[1e-3,2e-3],'time':[1e-6,2e-6,3e-6]})

testArray = cached_ndarray.cNDarray(values=values)
checkAnswer('ND by value, point, sample 0',testArray.values[0,0],1.0)
checkAnswer('ND by value, point, sample 1',testArray.values[1,0],11.0)
checkAnswer('ND by value, point, sample 2',testArray.values[2,0],21.0)

checkAnswer('ND by value, hist, sample 0, time 0 (access index)',testArray.values[0,1][0],2.0)
checkAnswer('ND by value, hist, sample 0, time 0 (access label)',testArray.values[0,1].loc[dict(time=1e-6)],2.0)
checkAnswer('ND by value, hist, sample 1, time 1',testArray.values[1,1][1],12.1)
checkAnswer('ND by value, hist, sample 2, time 2',testArray.values[2,1][2],22.2)

checkAnswer('ND by value, nd, sample 0, time 0, location 0 (access index)',testArray.values[0,2][0,0], 3.00)
checkAnswer('ND by value, nd, sample 0, time 0, location 0 (access label)',testArray.values[0,2].loc[dict(time=1e-6,space=1e-3)], 3.00)
checkAnswer('ND by value, nd, sample 1, time 0, location 1',testArray.values[1,2][0,1],13.01)
checkAnswer('ND by value, nd, sample 2, time 1, location 2',testArray.values[2,2][1,2],23.12)



#################
# LIST OF LISTS #
#################
# default constructor
testArray = cached_ndarray.listOfLists(width=3)
checkAnswer('initial width',testArray.shape[1],3)
checkAnswer('initial size',testArray.size,0)
checkAnswer('initial len',len(testArray),0)

vals = [ [1.0,
          xr.DataArray([ 2.0, 2.1, 2.2],dims=['time'],coords={'time':[1e-6,2e-6,3e-6]}),
          xr.DataArray([[ 3.00, 3.01, 3.02],[ 3.10, 3.11, 3.12]],dims=['space','time'],coords={'space':[1e-3,2e-3],'time':[1e-6,2e-6,3e-6]})
         ]
       ]
testArray.append(vals)
checkAnswer('NList append, point',testArray.values[0][0],1.0)
checkAnswer('NList append, hist, time 0',testArray.values[0][1][0],2.0)
checkAnswer('NList append, nd, time 0, location 0',testArray.values[0][2][0,0], 3.00)

# TODO values construction

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
