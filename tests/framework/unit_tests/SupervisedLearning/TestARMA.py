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
  This Module performs Unit Tests for the DataSet class.
  It can not be considered part of the active code but of the regression test system
"""

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import sys, os
import pickle as pk
import numpy as np
import copy

# find location of crow, message handler
frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4+['framework'])))
sys.path.append(frameworkDir)

from utils.utils import find_crow
find_crow(frameworkDir)
import MessageHandler

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug', 'callerLength':10, 'tagLength':10})

from SupervisedLearning import ARMA

print('Module undergoing testing:')
print(ARMA)
print('')

# seed the randomness
np.random.seed(42)

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
      if isinstance(val,(float,int)):
        pres = checkFloat('',val,second[key][0],tol,update=False)
      elif isinstance(val,(str,unicode)):
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

def createSignal(start,stop,periods,magnitudes=None):
  """
    Creates a deterministic signal with Fourier trends.
    @ In start, float, start time
    @ In stop, float, stop time
    @ In, periods, list, squared cosine waves to include
    @ In, magnitudes, list, optional, weights for each period (1:1 mapping)
    @ Out, signal, np.array, signal
  """
  if magnitudes is None:
    magnitudes = np.ones(len(periods))
  ts = np.linspace(start,stop,stop-start+1)
  ys = np.zeros(len(ts))
  for p,period in enumerate(periods):
    ys += magnitudes[p] * np.cos(ts*np.pi/period)**2
  return ys

def createBackCorrelatedNoise(length,lag):
  """
    Creates noise with a history
  """
  # base noise
  noise = np.random.rand(length) - 0.5
  # add lag
  for i in range(length):
    for j in range( min(i,lag) ):
      noise[i] += noise[i-j]/float(j+1)
  return noise

def createHistory(start,stop,periods,lag):
  """
    Creates full signal: Fourier + correlated noise
  """
  signal = createSignal(start,stop,periods)
  noise = createBackCorrelatedNoise(stop-start+1,lag)
  return signal+noise

######################################
#            CONSTRUCTION            #
######################################
# test initialization
kwargs = {'Target'  :'y,Time',
          'Features':'x',
          'Pmax':2,
          'Pmin':2,
          'Qmax':1,
          'Qmin':1}
arma = ARMA(mh,**kwargs)
# lag properties
for var in ['Pmax','Pmin','Qmax','Qmin']:
  checkSame('Set {}'.format(var),getattr(arma,var),kwargs[var])

######################################
#                CDF                 #
######################################
# test fitting of CDF to distributions

# start with single history
hist = createHistory(0,100,[10,20],2)
hist = np.array(zip(hist))
# check parameters
cdfParams = arma._generateCDF(hist)
checkFloat('CDF param min',cdfParams['CDFMin'],0.029702970297)
checkFloat('CDF param binsMax',cdfParams['binsMax'],2.78146536815)
checkFloat('CDF param binsMin',cdfParams['binsMin'],-0.877012421555)
correct = [0.02970297, 0.02970297, 0.11881188, 0.32673267, 0.55445545, 0.75247525, 0.92079208, 0.98019802, 1.]
checkArray('CDF param values',cdfParams['CDF'],correct,float,tol=1e-8)
correct = [-0.87701242, -0.4197027, 0.03760703, 0.49491675, 0.95222647, 1.4095362,1.86684592, 2.32415564, 2.78146537]
checkArray('CDF param bins',cdfParams['bins'],correct,float,tol=1e-8)

# check sampling, inverting
samples = 10
cdf = np.zeros(samples)
inv = np.zeros(samples)
xs = np.linspace(-1,3,samples)
vs = np.linspace(0,1,samples)
for i in range(samples):
  cdf[i] = arma._getCDF(cdfParams,xs[i])
  inv[i] = arma._getInvCDF(cdfParams,vs[i])
correct = [0.02970297, 0.02970297, 0.08983347, 0.25326703, 0.46758666,
           0.67136637, 0.84711436, 0.95252288, 0.99021787, 1.]
checkArray('CDF samples',cdf,correct,float,tol=1e-8)
correct = [-0.87701242, -0.04143416, 0.26505207, 0.50943452, 0.7313039,
            0.95476708,  1.21136865, 1.4782821,  1.62125366, 2.78146537]
checkArray('CDF inv samples',inv,correct,float,tol=1e-8)


# TODO add more unit tests to cover methods

print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.TestARMA</name>
    <author>talbpaul</author>
    <created>2018-05-09</created>
    <classesTested>SupervisedLearning.ARMA</classesTested>
    <description>
       This test is a Unit Test for the ARMA class.
    </description>
  </TestInfo>
"""
