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
  This Module performs Unit Tests for the TSA.Fourier class.
  It can not be considered part of the active code but of the regression test system
"""
import os
import sys
import copy
import numpy as np

np.random.seed(42)

# add RAVEN to path
frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)] + [os.pardir]*4 + ['framework'])))
if frameworkDir not in sys.path:
  sys.path.append(frameworkDir)

from utils.utils import find_crow
find_crow(frameworkDir)

from utils import xmlUtils, randomUtils

from TSA import Wavelet

plot = False

print('Module undergoing testing:')
print(Wavelet)
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
def createWaveletXML(targets, family):
  """
    Return Wavelet XML node for RAVEN input.

    @ In, targets, list[str], list of strings describing targets for current RAVEN run.
    @ In, family, str, the family of wavelet to use.
    @ Out, xml, xml.etree.ElementTree.Element, new node
  """
  xml = xmlUtils.newNode('Wavelet', attrib={'target':','.join(targets)})
  xml.append(xmlUtils.newNode('family', text=family))
  return xml

def createFromXML(xml):
  """
    Return Wavelet TSA object.

    @ In, xml, xml.etree.ElementTree.Element, RAVEN input xml.
    @ Out, wavelet, TSA.Wavelet, Wavelet object.
  """
  wavelet = Wavelet.Wavelet()
  inputSpec = wavelet.getInputSpecification()()
  inputSpec.parseNode(xml)
  wavelet.handleInput(inputSpec)
  return wavelet

def createWavelet(targets, family):
  """
    Return regression object.

    @ In, targets, list[str], list of string describing targets.
    @ In, degree, int, the degree of polynomial to fit.
    @ Out, wavelet, TSA.Wavelet, Wavelet object.
  """
  xml = createWaveletXML(targets, family)
  wavelet = createFromXML(xml)
  return wavelet

##########################################
# Discrete Wavelet Transform Simple Case #
##########################################
targets = ['A']
pivot = np.linspace(0, 8, 8)
N = len(pivot)
true_a = [5.65685425,  7.39923721,  0.22414387,  3.33677403,  7.77817459]
true_d = [-2.44948974, -1.60368225, -4.44140056, -0.41361256, 1.22474487]
titles = ['Simple Wavelet Transform Approximation Coefficients',
          'Simple Wavelet Transform Details Coefficients',]

signal = [3, 7, 1, 1, -2, 5, 4, 6]
signals = np.zeros((N, 1))
signals[:, 0] = signal

transform = createWavelet(targets, family='db2')
settings = {'family': 'db2'}
settings = transform.setDefaults(settings)
params = transform.characterize(signals, pivot, targets, settings)
check = params['A']['results']

for real_a, pred_a in zip(true_a, check['coeff_a']):
  checkFloat(titles[0], real_a, pred_a, tol=1e-8)

for real_d, pred_d in zip(true_d, check['coeff_d']):
  checkFloat(titles[1], real_d, pred_d, tol=1e-8)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.unit_tests.TSA.Wavelet</name>
    <author>dylanjm</author>
    <created>2021-02-24</created>
    <classesTested>TSA.Wavelet</classesTested>
    <description>
       This test is a Unit Test for the Wavelet TimeSeriesAnalyzer classes.
    </description>
  </TestInfo>
"""
