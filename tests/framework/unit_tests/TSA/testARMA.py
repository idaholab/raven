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

from TSA import ARMA

plot = False

print('Module undergoing testing:')
print(ARMA)
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
def createARMAXML(targets, P, Q):
  xml = xmlUtils.newNode('ARMA', attrib={'target':','.join(targets)})
  xml.append(xmlUtils.newNode('SignalLag', text=f'{P}'))
  xml.append(xmlUtils.newNode('NoiseLag', text=f'{Q}'))
  return xml

def createFromXML(xml):
  arma = ARMA()
  inputSpec = ARMA.getInputSpecification()()
  inputSpec.parseNode(xml)
  arma.handleInput(inputSpec)
  return arma

def createARMA(targets, P, Q):
  xml = createARMAXML(targets, P, Q)
  arma = createFromXML(xml)
  return arma

def createARMASignal(slags, nlags, pivot, noise=None, intercept=0, plot=False):
  if plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
  signal = np.zeros(len(pivot)) + intercept
  if noise is None:
    noise = np.random.normal(loc=0, scale=1, size=len(pivot))
  signal += noise
  # moving average: random noise lag
  for q, theta in enumerate(nlags):
    signal[q+1:] += theta * noise[:-(q+1)]
  # autoregressive: signal lag
  for t, time in enumerate(pivot):
    for p, phi in enumerate(slags):
      if t > p:
        signal[t] += phi * signal[t - p - 1]
  if plot:
    ax.plot(pivot, noise, 'k:')
    ax.plot(pivot, signal, 'g.-')
    plt.show()
  return signal, noise

###################
#  Simple         #
###################
# generate signal
targets = ['A'] #, 'B', 'C']
pivot = np.linspace(0, 100, 1000)
N = len(pivot)
smoothing_order = 1

slags = [0.4, 0.2]
nlags = [0.3, 0.2, 0.1]
order = [len(slags), 0, len(nlags)]
signalA, noise = createARMASignal(slags, nlags, pivot, plot=plot)

signals = np.zeros((len(pivot), 1))
signals[:, 0] = signalA

##########
# Simplest reasonable case
#
arma = createARMA(targets, 2, 3)
settings = {'P': 2, 'Q': 3,
            'gaussianize': False,
            'seed': 42}
settings = arma.setDefaults(settings)
params = arma.characterize(signals, pivot, targets, settings)
check = params['A']['arma']
# Note these are WAY OFF! They should match slags and nlags above.
# I don't know how to convince it to get
# any closer without "cheating" (giving it hints it shouldn't know about).
# If we can find any way to force this to behave better, that would be great.
# it seems like perhaps the likelihood fitter is not going to cooperate though.
okay_ar = [-0.03664183847944618, 0.46691996180943424]
okay_ma = [0.7333673476702858, 0.23819605887929196, 0.2293730352216328]
checkFloat('Simple ARMA intercept', 0.07723188355891732, check['const'], tol=1e-3)
checkArray('Simple ARMA AR', okay_ar, check['ar'], float, tol=1e-3)
checkArray('Simple ARMA MA', okay_ma, check['ma'], float, tol=1e-3)
checkFloat('Simple ARMA variance', 0.9532563046953576, check['var'], tol=1e-3)
# residual
# XXX FIXME WIP TODO
# r = params['A']['arma']['results']
# p = r.predict()
# f = r.fittedvalues
# res = r.resid
# freq, bins = np.histogram(res, bins=32)
# #p = np.hstack([p[1:], np.atleast_1d(p[0])]) # left-shift
# #resid = signalA - p
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(0.5*(bins[1:]+bins[:-1]), freq, '.-')
# #ax.plot(f, res, '.')
# #ax.plot(pivot, signalA, 'o-')
# #ax.plot(pivot, p, '+-')
# #ax.plot(pivot, f, 'x-')
# #ax.plot(pivot, res, '.:')
# plt.show()

checkTrue("arma can generate", arma.canGenerate())
checkTrue("arma can characterize", arma.canCharacterize())

# predict
np.random.seed(42) # forces MLE in statsmodels to be deterministic
new = arma.generate(params, pivot, settings)[:, 0]

# spot check a few values -> could we check full arrays?
checkFloat('Simple generate 0', -1.0834098074509528, new[0], tol=1e-6)
checkFloat('Simple generate 250', -3.947707011147049, new[250], tol=1e-6)
checkFloat('Simple generate 500', -1.4304498185153571, new[500], tol=1e-6)
checkFloat('Simple generate 999', -1.7825760423361088, new[999], tol=1e-6)
# now do it again, but set the params how we want to
params['A']['arma']['const'] = 0
params['A']['arma']['AR'] = slags
params['A']['arma']['MA'] = nlags
params['A']['arma']['var'] = 1
np.random.seed(42) # forces MLE in statsmodels to be deterministic
new = arma.generate(params, pivot, settings)[:, 0]
checkFloat('Simple picked 0', 2.3613260219896035, new[0], tol=1e-6)
checkFloat('Simple picked 250', -1.4007530275511393, new[250], tol=1e-6)
checkFloat('Simple picked 500', 0.7956991243820065, new[500], tol=1e-6)
checkFloat('Simple picked 999', 0.7196164370698425, new[999], tol=1e-6)

##########
# Gaussianize, but we don't technically need to.
# That is, noise is already ~N(0, 1), but we go through the denormalization anyway
#
settings = {'P': 2, 'Q': 3,
            'gaussianize': True,
            'seed': 42}
settings = arma.setDefaults(settings)
params = arma.characterize(signals, pivot, targets, settings)
# These are a little different from the non-Gaussianize above, but pretty close (kind of).
# Given the numerical nature of the empirical CDF, maybe not too bad.
okay_ar = [-0.1288380832279767, 0.5286049589896539]
okay_ma = [0.7722899504423865, 0.18289761662693169, 0.20950559786741266]
checkArray('Gaussian ARMA AR', okay_ar, params['A']['arma']['ar'], float, tol=1e-3)
checkArray('Gaussian ARMA MA', okay_ma, params['A']['arma']['ma'], float, tol=1e-3)
np.random.seed(42) # forces MLE in statsmodels to be deterministic
new = arma.generate(params, pivot, settings)[:, 0]
checkFloat('Simple denorm 0', -1.459305773140902, new[0], tol=1e-6)
checkFloat('Simple denorm 250', 2.051286365253135, new[250], tol=1e-6)
checkFloat('Simple denorm 500', -0.5047179383332892, new[500], tol=1e-6)
checkFloat('Simple denorm 999', 1.3200315405820204, new[999], tol=1e-6)

print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.unit_tests.TSA.Fourier</name>
    <author>talbpaul</author>
    <created>2021-01-05</created>
    <classesTested>TSA.Fourier</classesTested>
    <description>
       This test is a Unit Test for the Fourier TimeSeriesAnalyzer classes.
    </description>
  </TestInfo>
"""
