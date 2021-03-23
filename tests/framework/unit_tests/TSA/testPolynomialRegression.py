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
  This Module performs Unit Tests for the TSA.PolynomialRegression class.
  It can not be considered part of the active code but of the regression test system
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# add RAVEN to path
frameworkDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)] + [os.pardir]*4 + ['framework'])))
if frameworkDir not in sys.path:
  sys.path.append(frameworkDir)

from utils.utils import find_crow
find_crow(frameworkDir)

from utils import xmlUtils
from TSA import PolynomialRegression as PR

plot = False

print(f"\nModule undergoing testing:\n{PR}\n")

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
def createRegressionXML(targets, degree):
  """
    Return regression XML node for RAVEN input.

    @ In, targets, list[str], list of strings describing targets for current RAVEN run.
    @ In, degree, int, the degree of polynomial to fit.
    @ Out, xml, xml.etree.ElementTree.Element, new node
  """
  xml = xmlUtils.newNode('Regression', attrib={'target':','.join(targets)})
  xml.append(xmlUtils.newNode('degree', text=f'{degree}'))
  return xml

def createFromXML(xml):
  """
    Return PolynomialRegression TSA object.

    @ In, xml, xml.etree.ElementTree.Element, RAVEN input xml.
    @ Out, regression, TSA.PolynomialRegression, Regression object.
  """
  regression = PR.PolynomialRegression()
  inputSpec = PR.PolynomialRegression.getInputSpecification()()
  inputSpec.parseNode(xml)
  regression.handleInput(inputSpec)
  return regression

def createRegression(targets, degree):
  """
    Return regression object.

    @ In, targets, list[str], list of string describing targets.
    @ In, degree, int, the degree of polynomial to fit.
    @ Out, regression, TSA.PolynomialRegression, Regression object.
  """
  xml = createRegressionXML(targets, degree)
  regression = createFromXML(xml)
  return regression

###################
#  Simple         #
###################
# generate signal
targets = ['A'] #, 'B', 'C']
pivot = np.linspace(0, 100, 100)
N = len(pivot)
okay_coefs = [1.0, 2.0, 3.0]
coef_titles = ['Simple Polynomial Regression Intercept',
               'Simple Polynomial Regression Coeff 1',
               'Simple Polynomial Regression Coeff 2',]

signalA = okay_coefs[0] + (okay_coefs[1] * pivot) + (okay_coefs[2] * (pivot**2))
signals = np.zeros((N, 1))
signals[:, 0] = signalA

model = createRegression(targets, 2)
settings = {'degree': 2}
settings = model.setDefaults(settings)
params = model.characterize(signals, pivot, targets, settings)
check = params['A']['model']

for title, real, pred in zip(coef_titles, okay_coefs, check):
  checkFloat(title, real, check[pred], tol=1e-1)

checkTrue("model can generate", model.canGenerate())
checkTrue("model can characterize", model.canCharacterize())

############################
# Simple w/ Random Noise   #
############################
signalA += np.random.normal(0, 1, 100)
signals = np.zeros((N, 1))
signals[:, 0] = signalA

model = createRegression(targets, 2)
settings = {'degree': 2}
settings = model.setDefaults(settings)
params = model.characterize(signals, pivot, targets, settings)
check = params['A']['model']

for title, real, pred in zip(coef_titles, okay_coefs, check):
  checkFloat(title, real, check[pred], tol=1e-1)

################
# Complex Case #
################
okay_coefs += [4.0]
coef_titles += ['Simple Polynomial Regression Coeff 3']
signalB = okay_coefs[0] + (okay_coefs[1] * pivot) + (okay_coefs[2] * (pivot**2)) + (okay_coefs[3] * (pivot**3))
signals = np.zeros((N, 1))
signals[:, 0] = signalB

model = createRegression(targets, 3)
settings = {'degree': 3}
settings = model.setDefaults(settings)
params = model.characterize(signals, pivot, targets, settings)
check = params['A']['model']

for title, real, pred in zip(coef_titles, okay_coefs, check):
  checkFloat(title, real, check[pred], tol=1e-1)

################################
# Complex Case w/ Random Noise #
################################
################
# Complex Case #
################
signalB += np.random.normal(0, 1, 100)
signals = np.zeros((N, 1))
signals[:, 0] = signalB

model = createRegression(targets, 3)
settings = {'degree': 3}
settings = model.setDefaults(settings)
params = model.characterize(signals, pivot, targets, settings)
check = params['A']['model']

for title, real, pred in zip(coef_titles, okay_coefs, check):
  checkFloat(title, real, check[pred], tol=1e-1)

print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.unit_tests.TSA.PolynomialRegression</name>
    <author>dylanjm</author>
    <created>2021-02-16</created>
    <classesTested>TSA.PolynomialRegression</classesTested>
    <description>
       This test is a Unit Test for the PolynomialRegression TimeSeriesAnalyzer classes.
    </description>
  </TestInfo>
"""
