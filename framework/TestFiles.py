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
  This Module performs Unit Tests for the Files class.
  It cannot be considered part of the active code but of the regression test system
"""

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys, os
import cPickle as pk
import numpy as np
import xml.etree.ElementTree as ET
frameworkDir = os.path.dirname(os.path.abspath(os.path.join(sys.argv[0])))

sys.path.append(frameworkDir)
from utils.utils import find_crow
from utils import xmlUtils

find_crow(frameworkDir)

import MessageHandler
mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug', 'callerLength':10, 'tagLength':10})

import Files
print('Module undergoing testing:')
print (Files )
print('')

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

def checkNone(comment,entry,update=True):
  """
    Checks if entry is None.
    @ In, comment, string, a comment printed out if it fails
    @ In, entry, object, entity to evaluate for None
    @ In, update, bool, optional, if True then will update results
    @ Out, res, bool, True if entry is None
  """
  res = entry is None
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking answer",comment,'|','"{}" is not None!'.format(entry))
      results["fail"] += 1
  return res

def checkFails(comment,errstr,function,update=True,args=None,kwargs=None):
  """
    Checks if entry is None.
    @ In, comment, string, a comment printed out if it fails
    @ In, errstr, str, message given by error to check against
    @ In, function, method, function to evaluate that is expected to fail
    @ In, update, bool, optional, if True then will update results
    @ In, args, list, additional arguments to pass to function
    @ In, kwargs, dict, additional keyword arguments to pass to function
    @ Out, res, bool, True if entry is None
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
    else:
      print("checking error",comment,'|',msg)
      results["fail"] += 1
  print(' ... end Error testing')
  print('')
  return res

# TODO test RAVENGenerated, File (base type), CSV, DynamicXMLOutput, and UserGenerated types
# These are covered in regression tests, but not yet in unit tests.

######################################
#          Static XML                #
######################################
# create it
static = Files.StaticXMLOutput()
static.messageHandler = mh
# start a new tree
static.newTree('TestRoot')
# test instance
checkTrue('Static root node type',isinstance(static.tree,ET.ElementTree))
# test tag
checkSame('Static root node name',static.tree.getroot().tag,'TestRoot')
# add some scalars
static.addScalar('FirstTarget','pi',3.14159)
static.addScalar('FirstTarget','e',6.18)
# check values
pi = xmlUtils.findPath(static.tree.getroot(),'FirstTarget/pi')
e = xmlUtils.findPath(static.tree.getroot(),'FirstTarget/e')
checkFloat('Static pi',float(pi.text),3.14159)
checkFloat('Static e',float(e.text),6.18)
# add a vector
static.addVector('SecondTarget','pi',{'e':0,'pi':1})
static.addVector('SecondTarget','e',{'e':1,'pi':0})
# check values
checkFloat('Static pi wrt pi',float(xmlUtils.findPath(static.tree.getroot(),'SecondTarget/pi/pi').text),1.0)
checkFloat('Static pi wrt e',float(xmlUtils.findPath(static.tree.getroot(),'SecondTarget/pi/e').text),0.0)
checkFloat('Static e wrt pi',float(xmlUtils.findPath(static.tree.getroot(),'SecondTarget/e/pi').text),0.0)
checkFloat('Static e wrt e',float(xmlUtils.findPath(static.tree.getroot(),'SecondTarget/e/e').text),1.0)
# test pickling
p = pk.dumps(static)
new = pk.loads(p)
# check all values on new
checkTrue('Static pk root node type',isinstance(static.tree,ET.ElementTree))
checkSame('Static pk root node name',static.tree.getroot().tag,'TestRoot')
pi = xmlUtils.findPath(new.tree.getroot(),'FirstTarget/pi')
e = xmlUtils.findPath(new.tree.getroot(),'FirstTarget/e')
checkFloat('Static pk pi',float(pi.text),3.14159)
checkFloat('Static pk e',float(e.text),6.18)
checkFloat('Static pk pi wrt pi',float(xmlUtils.findPath(new.tree.getroot(),'SecondTarget/pi/pi').text),1.0)
checkFloat('Static pk pi wrt e',float(xmlUtils.findPath(new.tree.getroot(),'SecondTarget/pi/e').text),0.0)
checkFloat('Static pk e wrt pi',float(xmlUtils.findPath(new.tree.getroot(),'SecondTarget/e/pi').text),0.0)
checkFloat('Static pk e wrt e',float(xmlUtils.findPath(new.tree.getroot(),'SecondTarget/e/e').text),1.0)

# TODO test writing to file


print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.unit_test_Files</name>
    <author>talbpaul</author>
    <created>2017-11-01</created>
    <classesTested>Files</classesTested>
    <description>
       This test is a Unit Test for the Files classes.
    </description>
  </TestInfo>
"""
