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
  This Module performs Unit Tests for the Distribution class.
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

import XrDataObject # FIXME
import MessageHandler

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug', 'callerLength':10, 'tagLength':10})

print('Module undergoing testing:')
print (XrDataObject )
print('')

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
    @ Out, res, bool, True if same
  """
  res = abs(value - expected) <= tol
  if update:
    if res:
      print("checking answer",comment,'|',value,"!=",expected)
      results["fail"] += 1
    else:
      results["pass"] += 1
  return res

def checkString(comment,value,expected,update=True):
  """
    This method is aimed to compare two strings
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ Out, res, bool, True if same
  """
  res = value == expected
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking answer",comment,'|',value,"!=",expected)
      results["fail"] += 1
  return res

def checkArray(comment,first,second,dtype,tol=1e-10,update=True):
  """
    This method is aimed to compare two arrays
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
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
      elif dtype == str:
        pres = checkString('',first[i],second[i],update=False)
      if not pres:
        print('checking answer',comment,'|','entry "{}" does not match: {} != {}'.format(i,first[i],second[i]))
        res = False
  if update:
    if res:
      results["pass"] += 1
    else:
      results["fail"] += 1
  return res

def checkNone(comment,entry,update=True):
  res = entry is None
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking answer",comment,'|','"{}" is not None!'.format(entry))
      results["fail"] += 1

def checkFails(comment,errstr,function,update=True,args=None,kwargs=None):
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
    res = checkString('',e.args[0],errstr,update=False)
    if not res:
      msg = 'Unexpected error message.  \n    Received: "{}"\n    Expected: "{}"'.format(e.args[0],errstr)
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking error",comment,'|',msg)
      results["fail"] += 1
  print(' ... end Error testing')
  print('')
  return res



#Test module methods #TODO
#print(Distributions.knownTypes())
#Test error
#try:
#  Distributions.returnInstance("unknown",'dud')
#except:
#  print("error worked")

xml = createElement('DataSet',attrib={'name':'test'})
xml.append(createElement('Input',text='a,b,c'))
xml.append(createElement('Output',text='x,y,z'))

# check construction
data = XrDataObject.DataSet()
# inputs, outputs
checkString('DataSet __init__ name',data.name,'DataSet')
checkString('DataSet __init__ print tag',data.printTag,'DataSet')
checkNone('DataSet __init__ _data',data._data)
checkNone('DataSet __init__ _collector',data._collector)

# check initialization
data._readMoreXML(xml)
data.messageHandler = mh
checkArray('DataSet __init__ inp',data._inputs,['a','b','c'],str)
checkArray('DataSet __init__ out',data._outputs,['x','y','z'],str)
checkArray('DataSet __init__ all',data._allvars,['a','b','c','x','y','z'],str)
checkNone('DataSet __init__ _data',data._data)
checkNone('DataSet __init__ _collector',data._collector)

# append some data to get started
rlz1 = {'a': 1.0,
        'b': 2.0,
        'c': xr.DataArray([3.0, 3.1, 3.2],dims=['time'],coords={'time':[3.1e-6,3.2e-6,3.3e-6]}),
        'x': 4.0,
        'y': xr.DataArray([5.0, 5.1, 5.2],dims=['time'],coords={'time':[5.1e-6,5.2e-6,5.3e-6]}),
       }
rlz2 = {'a' :11.0,
        'b': 12.0,
        'c': xr.DataArray([13.0, 13.1, 13.2],dims=['time'],coords={'time':[13.1e-6,13.2e-6,13.3e-6]}),
        'x': 14.0,
        'y': xr.DataArray([15.0, 15.1, 15.2],dims=['time'],coords={'time':[15.1e-6,15.2e-6,15.3e-6]}),
        'z': 16.0
       }
# test missing data
checkFails('DataSet addRealization err','Provided realization does not have all requisite values: \"z\"',data.addRealization,args=[rlz1])
rlz1['z'] = 6.0
# test appending
data.addRealization(rlz1)
# test contents after first append
print(data.getRealization(index=0))
data.addRealization(rlz2)


# check builtins
# check basic property getters

# check appending (add row)
# check add var (add column)
# check remove var (remove column)
# check remove sample (remove row)

# check slicing
# # var vals
# # realization vals
# # by meta
# check find-by-index
# check find-by-value (including find-by-meta)

# check write to CSV
# check write to netCDF
# check load from CSV
# check load from netCDF


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
