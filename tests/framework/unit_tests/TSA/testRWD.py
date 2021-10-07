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
  This Module performs Unit Tests for the TSA.RWD class.
  It can not be considered part of the active code but of the regression test system
"""
import os
import sys
import copy
import numpy as np
np.random.seed(42)

rng = np.random.default_rng(seed=42)

# add RAVEN to path
frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)] + [os.pardir]*4 + ['framework'])))
if frameworkDir not in sys.path:
  sys.path.append(frameworkDir)

from utils.utils import find_crow
find_crow(frameworkDir)
import numpy.linalg as LA

from utils import xmlUtils

from TSA import RWD

plot = False

print('Module undergoing testing:')
print(RWD)
print('')

results = {"pass":0,"fail":0}

def checkFloat(comment, value, expected, tol=1e-10, update=True):
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

def checkTrue(comment, res, update=True):
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

def checkSame(comment, value, expected, update=True):
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

def checkArray(comment, first, second, dtype, tol=1e-10, update=True):
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

def checkNone(comment, entry, update=True):
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

def checkFails(comment, errstr, function, update=True, args=None, kwargs=None):
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


######################################
#            CONSTRUCTION            #
######################################
def createRWDXML(targets, signatureWindowLength, featureIndex, sampleType):
  xml = xmlUtils.newNode('RWD', attrib={'target':','.join(targets)})
  xml.append(xmlUtils.newNode('signatureWindowLength', text=f'{signatureWindowLength}'))
  xml.append(xmlUtils.newNode('featureIndex', text=f'{featureIndex}'))
  xml.append(xmlUtils.newNode('sampleType', text=f'{sampleType}'))
  return xml

def createFromXML(xml):
  print(vars(RWD))
  rwd = RWD()
  
  inputSpec = rwd.getInputSpecification()()
  inputSpec.parseNode(xml)
  rwd.handleInput(inputSpec)
  return rwd

def createRWD(targets, signatureWindowLength, featureIndex, sampleType):
  xml = createRWDXML(targets, signatureWindowLength, featureIndex, sampleType)
  print('createRWDXML passed')
  rwd = createFromXML(xml)
  print('createFromXML passed')
  return rwd


###################
#  Simple         #
###################
# generate signal
targets = ['A'] #, 'B', 'C']
pivot = np.linspace(0, 100, 1000)
N = len(pivot) 
s = np.linspace(-10,10,1000)
time_series = 3.5*s+s**2+10


signals = np.zeros((len(pivot), 1))
signals[:, 0] = time_series

##########
# Simplest reasonable case
#
rwd = createRWD(targets, 105, 3,0)
settings = {'signatureWindowLength':105, 'featureIndex': 3, 'sampleType' : 0}
if 'signatureWindowLength' not in settings:
  print('signatureWindowLength is not in settings')
settings = rwd.setDefaults(settings)
params = rwd.characterize(signals, pivot, targets, settings)
check = params['A']



# code to generate answer
signatureWindowLength = 105
history = np.copy(time_series)
sampleLimit = len(history)-signatureWindowLength
windowNumber = sampleLimit//4
sampleIndex = np.random.randint(sampleLimit, size=windowNumber)
baseMatrix = np.zeros((signatureWindowLength, windowNumber))
for i in range(windowNumber):
  windowIndex = sampleIndex[i]
  baseMatrix[:,i] = np.copy(history[windowIndex:windowIndex+signatureWindowLength])

allwindowNumber = len(history)-signatureWindowLength+1
signatureMatrix = np.zeros((signatureWindowLength, allwindowNumber))

for i in range(allwindowNumber):
  signatureMatrix[:,i] = np.copy(history[i:i+signatureWindowLength])

baseMatrix = np.copy(signatureMatrix)
U,S,V = LA.svd(baseMatrix,full_matrices=False)
assert np.allclose(baseMatrix, np.dot(U, np.dot(np.diag(S), V)))


F = U.T @ signatureMatrix  
okay_basis = U[:,0]
okay_feature = F[0:3,0]


checkArray('Simple RWD Basis', okay_basis, check['uVec'][:,0], float, tol=1e-3)
checkArray('Simple RWD Features', okay_feature, check['Feature'][0:3,0], float, tol=1e-3)
checkTrue("RWD can characterize", rwd.canCharacterize())

print(results)

sys.exit(results["fail"])

