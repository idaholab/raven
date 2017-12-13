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
  This Module performs Unit Tests for Hierarchal data objects
  It can not be considered part of the active code but of the regression test system
"""

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import xml.etree.ElementTree as ET
import sys, os
import pickle as pk
import numpy as np
import xarray as xr
frameworkDir = os.path.dirname(os.path.abspath(os.path.join(sys.argv[0],'..')))

sys.path.append(frameworkDir)
from utils.utils import find_crow

find_crow(frameworkDir)

import XDataSet
import MessageHandler

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'silent', 'callerLength':10, 'tagLength':10})

def createElement(tag,attrib=None,text=None):
  """
    Method to create a dummy xml element readable by the distribution classes
    @ In, tag, string, the node tag
    @ In, attrib, dict, optional, the attribute of the xml node
    @ In, text, str, optional, the dict containig what should be in the xml text
  """
  if attrib is None:
    attrib = {}
  if text is None:
    text = ''
  element = ET.Element(tag,attrib)
  element.text = text
  return element

results = {"pass":0,"fail":0}

def checkFloat(comment,value,expected,tol=1e-10,update=True):
  """
    This method is aimed to compare two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  if np.isnan(value) and np.isnan(expected):
    res = True
  elif np.isnan(value) or np.isnan(expected):
    res = False
  else:
    res = abs(value - expected) <= tol
  if update:
    if not res:
      print("checking float",comment,'|',value,"!=",expected)
      results["fail"] += 1
    else:
      results["pass"] += 1
  return res

def checkTrue(comment,res,update=True):
  """
    This method is a pass-through for consistency and updating
    @ In, comment, string, a comment printed out if it fails
    @ In, res, bool, the tested value
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if test
  """
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking bool",comment,'|',res,'is not True!')
      results["fail"] += 1
  return res

def checkSame(comment,value,expected,update=True):
  """
    This method is aimed to compare two identical things
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = value == expected
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking string",comment,'|',value,"!=",expected)
      results["fail"] += 1
  return res

def checkArray(comment,first,second,dtype,tol=1e-10,update=True):
  """
    This method is aimed to compare two arrays
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = True
  if len(first) != len(second):
    res = False
    print("checking answer",comment,'|','lengths do not match:',len(first),len(second))
  else:
    for i in range(len(first)):
      if dtype == float:
        pres = checkFloat('',first[i],second[i],tol,update=False)
      elif dtype in (str,unicode):
        pres = checkSame('',first[i],second[i],update=False)
      if not pres:
        print('checking array',comment,'|','entry "{}" does not match: {} != {}'.format(i,first[i],second[i]))
        res = False
  if update:
    if res:
      results["pass"] += 1
    else:
      results["fail"] += 1
  return res

def checkRlz(comment,first,second,tol=1e-10,update=True,skip=None):
  """
    This method is aimed to compare two realization
    @ In, comment, string, a comment printed out if it fails
    @ In, first, dict, the first dict, the "calculated" value -> should be as obtained from the data object
    @ In, second, dict, the second dict, the "expected" value -> should be as a realization submitted
    @ In, tol, float, optional, the tolerance
    @ In, update, bool, optional, if False then don't update results counter
    @ In, skip, list, optional, keywords not to check
    @ Out, res, bool, True if same
  """
  if skip is None:
    skip = []
  res = True
  if abs(len(first) - len(second)) > len(skip):
    res = False
    print("checking answer",comment,'|','lengths do not match:',len(first),len(second))
  else:
    for key,val in first.items():
      if key in skip:
        continue
      if isinstance(val,(float,int)) and not isinstance(val,bool):
        pres = checkFloat('',val,second[key][0],tol,update=False)
      elif isinstance(val,(str,unicode,bool,np.bool_)):
        pres = checkSame('',val,second[key][0],update=False)
      elif isinstance(val,np.ndarray):
        if isinstance(val[0],(float,int)):
          pres = (val - second[key]).sum()<1e-20 #necessary due to roundoff
        else:
          pres = val == second[key]
      elif isinstance(val,xr.DataArray):
        if isinstance(val.item(0),(float,int)):
          pres = (val - second[key]).sum()<1e-20 #necessary due to roundoff
        else:
          pres = val.equals(second[key])
      else:
        raise TypeError(type(val))
      if not pres:
        print('checking dict',comment,'|','entry "{}" does not match: {} != {}'.format(key,first[key],second[key]))
        res = False
  if update:
    if res:
      results["pass"] += 1
    else:
      results["fail"] += 1
  return res

def checkNone(comment,entry,update=True):
  """
    Checks if entry is None.
    @ In, comment, string, a comment printed out if it fails
    @ In, entry, object, to test if against None
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if None
  """
  res = entry is None
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking answer",comment,'|','"{}" is not None!'.format(entry))
      results["fail"] += 1

def checkFails(comment,errstr,function,update=True,args=None,kwargs=None):
  """
    Checks if expected error occurs
    @ In, comment, string, a comment printed out if it fails
    @ In, errstr, str, expected fail message
    @ In, function, method, method to run to test for failure
    @ In, update, bool, optional, if False then don't update results counter
    @ In, args, list, arguments to pass to function
    @ In, kwargs, dict, keyword arguments to pass to function
    @ Out, res, bool, True if failed as expected
  """
  print('Error testing ...')
  if args is None:
    args = []
  if kwargs is None:
    kwargs = {}
  try:
    function(*args,**kwargs)
    res = False
    msg = 'Function call did not error!'
  except Exception as e:
    res = checkSame('',e.args[0],errstr,update=False)
    if not res:
      msg = 'Unexpected error message.  \n    Received: "{}"\n    Expected: "{}"'.format(e.args[0],errstr)
  if update:
    if res:
      results["pass"] += 1
      print(' ... end Error testing (PASSED)')
    else:
      print("checking error",comment,'|',msg)
      results["fail"] += 1
      print(' ... end Error testing (FAILED)')
  print('')
  return res


def formatRealization(rlz):
  """
    Converts types of each input.
    @ In, rlz, dict, var:val
    @ Out, rlz, dict, formatted
  """
  for k,v in rlz.items():
    rlz[k] = np.atleast_1d(v)

######################################
#            CONSTRUCTION            #
######################################
xml = createElement('DataSet',attrib={'name':'test'})
xml.append(createElement('Input',text='a,'))
xml.append(createElement('Output',text='b'))
xml.append(createElement('Index',attrib={'var':'time'},text='b'))
# construct
data = XDataSet.DataSet()
# initialization
data.messageHandler = mh
mh.verbosity = 'debug'
data._readMoreXML(xml)
# register expected meta
data.addExpectedMeta(['prefix','RAVEN_parentID','RAVEN_isEnding'])

#### Explanation of Test ####
# We're going to simulate the storage of a heirarchical data tree as follows:
#
#                  |-1_1_1-
#         |-1_1----|
#         |        |-1_1_2-
# -1------|
#         |        |-1_2_1-
#         |-1_2----|
#                  |-1_2_2-
#
# Level: 1     2       3
# with variable "a" as scalar (and the cause of the splits), and "b" as time-dependent
# The meta variables are RAVEN_HierID (as shown in the tree with underscores) and RAVEN_HierLevel
# (as shown by Level underneath).  On each branch, "a" stays the same on the _1 branch, and doubles
# on the _2 branch.
rlz1 = {              'a':np.array([1.0        ]),
                      'b':np.array([0.1,0.2,0.3]),
                   'time':np.array([1.0,2.0,3.0])*1e-6,
                 'prefix':np.array(["1"        ]),
         'RAVEN_parentID':np.array([None       ]),
         'RAVEN_isEnding':np.array([True       ])}

rlz1_1 = {            'a':np.array([1.0        ]),
                      'b':np.array([0.4,0.5,0.6]),
                   'time':np.array([4.0,5.0,6.0])*1e-6,
                 'prefix':np.array(["2"        ]),
         'RAVEN_parentID':np.array(["1"        ]),
         'RAVEN_isEnding':np.array([True       ])}

rlz1_2 = {            'a':np.array([2.0        ]),
                      'b':np.array([0.7,0.8,0.9]),
                   'time':np.array([7.0,8.0,9.0])*1e-6,
                 'prefix':np.array(["3"        ]),
         'RAVEN_parentID':np.array(["1"        ]),
         'RAVEN_isEnding':np.array([True       ])}

rlz1_1_1 = {          'a':np.array([1.0        ]),
                      'b':np.array([1.0,1.1,1.2]),
                   'time':np.array([1.0,1.1,1.2])*1e-5,
                 'prefix':np.array(["4"        ]),
         'RAVEN_parentID':np.array(["2"        ]),
         'RAVEN_isEnding':np.array([True       ])}

rlz1_1_2 = {          'a':np.array([2.0        ]),
                      'b':np.array([1.3,1.4,1.5]),
                   'time':np.array([1.3,1.4,1.5])*1e-5,
                 'prefix':np.array(["5"        ]),
         'RAVEN_parentID':np.array(["2"        ]),
         'RAVEN_isEnding':np.array([True       ])}

rlz1_2_1 = {          'a':np.array([2.0        ]),
                      'b':np.array([1.6,1.7,1.8]),
                   'time':np.array([1.6,1.7,1.8])*1e-5,
                 'prefix':np.array(["6"        ]),
         'RAVEN_parentID':np.array(["3"        ]),
         'RAVEN_isEnding':np.array([True       ])}

rlz1_2_2 = {          'a':np.array([4.0        ]),
                      'b':np.array([1.9,2.0,2.1]),
                   'time':np.array([1.9,2.0,2.1])*1e-5,
                 'prefix':np.array(["7"        ]),
         'RAVEN_parentID':np.array(["3"        ]),
         'RAVEN_isEnding':np.array([True       ])}

data.addRealization(rlz1)
data.addRealization(rlz1_1)
data.addRealization(rlz1_2)
data.addRealization(rlz1_1_1)
# check mid-creation, calling asDataset
## this assures that we can work with the collector OR the data equally well
data.asDataset()
endings = data._getPathEndings()
#for e,end in enumerate(endings):
checkRlz('Path early endings[0]',endings[0],rlz1_2,skip='time')
checkRlz('Path early endings[1]',endings[1],rlz1_1_1,skip='time')
paths = data._generateHierPaths()
full = data._constructHierPaths()


data.addRealization(rlz1_1_2)
data.addRealization(rlz1_2_1)
data.addRealization(rlz1_2_2)
# make a dataset TODO for less slowdown, see if we can do it with the Collector AND the data

# check endings
endings = data._getPathEndings()
checkRlz('Path endings[0]',endings[0],rlz1_1_2,skip='time')
checkRlz('Path endings[1]',endings[1],rlz1_2_1,skip='time')
checkRlz('Path endings[2]',endings[2],rlz1_2_2,skip='time')
checkRlz('Path endings[3]',endings[3],rlz1_1_1,skip='time')

# check paths
paths = data._generateHierPaths()
checkArray('Path paths[0]',paths[3],['1','2','4'],str)
checkArray('Path paths[1]',paths[0],['1','2','5'],str)
checkArray('Path paths[2]',paths[1],['1','3','6'],str)
checkArray('Path paths[3]',paths[2],['1','3','7'],str)

# get fully constructed path data
full = data._constructHierPaths()
# check some select data
checkArray('Path full[0] RSID',full[0]['RAVEN_sample_ID'].values,[0,1,3],float)
checkArray('Path full[1] RSID',full[1]['RAVEN_sample_ID'].values,[0,1,4],float)
checkArray('Path full[2] RSID',full[2]['RAVEN_sample_ID'].values,[0,2,5],float)
checkArray('Path full[3] RSID',full[3]['RAVEN_sample_ID'].values,[0,2,6],float)
checkArray('Path full[0]',full[0]['prefix'],['1','2','4'],str)
checkArray('Path full[1]',full[1]['prefix'],['1','2','5'],str)
checkArray('Path full[2]',full[2]['prefix'],['1','3','6'],str)
checkArray('Path full[3]',full[3]['prefix'],['1','3','7'],str)
checkArray('Path full[0] a',full[0]['a'].values,[1,1,1],float)
checkArray('Path full[1] a',full[1]['a'].values,[1,1,2],float)
checkArray('Path full[2] a',full[2]['a'].values,[1,2,2],float)
checkArray('Path full[3] a',full[3]['a'].values,[1,2,4],float)
for f,fl in enumerate(full):
  checkArray('Path full[{}] isEnding'.format(f),fl['RAVEN_isEnding'],[False,False,True],str)

print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.test_datasets</name>
    <author>talbpaul</author>
    <created>2017-10-20</created>
    <classesTested>DataSet</classesTested>
    <description>
       This test is a Unit Test for the DataSet classes.
    </description>
  </TestInfo>
"""
