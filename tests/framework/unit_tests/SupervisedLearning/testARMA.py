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
  This Module performs Unit Tests for the ARMA class.
  It can not be considered part of the active code but of the regression test system
"""
import xml.etree.ElementTree as ET
import sys, os
from scipy import stats
import pickle as pk
import numpy as np
import pandas as pd

# find location of crow, message handler
ravenDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4)))

sys.path.append(ravenDir)
frameworkDir = os.path.join(ravenDir, 'framework')

from ravenframework.utils.utils import find_crow
find_crow(frameworkDir)
from ravenframework.utils import randomUtils

from ravenframework import MessageHandler

# message handler
mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug', 'callerLength':10, 'tagLength':10})

# input specs come mostly from the Models.ROM
from ravenframework.Models import ROM

# find location of ARMA
from ravenframework.SupervisedLearning import ARMA

print('Module undergoing testing:')
print(ARMA)
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


######################################
#            CONSTRUCTION            #
######################################
def createARMAXml(targets, pivot, p, q, fourier=None):
  if fourier is None:
    fourier = []
  xml = createElement('ROM',attrib={'name':'test', 'subType':'ARMA'})
  xml.append(createElement('Target',text=','.join(targets+[pivot])))
  xml.append(createElement('Features',text='scaling'))
  xml.append(createElement('pivotParameter',text=pivot))
  xml.append(createElement('P',text=str(p)))
  xml.append(createElement('Q',text=str(q)))
  if len(fourier):
    xml.append(createElement('Fourier',text=','.join(str(f) for f in fourier)))
  return xml

def createFromXML(xml):
  inputSpec = ROM.getInputSpecification(xml)
  rom = ROM()
  rom._readMoreXML(xml)
  arma = rom.supervisedContainer[0]
  return rom, arma

def createARMA(targets, pivot, p, q, fourier=None):
  xml = createARMAXml(targets, pivot, p, q, fourier)
  rom, arma = createFromXML(xml)
  return rom, arma

rom, arma = createARMA(['a','b'], 't', 6, 3, [86400,43200])

# TODO confirmation testing for correct construction

#############################################
#            CDF, STATS OPERATIONS          #
#############################################

def makeCDF(data, bins=70):
  if bins is None:
    bins = int(np.sqrt(len(data)+0.5))
  # actually makes pdf and cdf, returns both
  counts, edges = np.histogram(data, bins=bins, density=False)
  counts = np.array(counts) / float(len(data))
  return (edges, counts*bins), (edges, np.insert(np.cumsum(counts),0,0))

def plotCDF(edges, bins, ax, label, color, alpha=1.0):
  for e,edge in enumerate(edges[:-1]):
    if e == 0:
      label = label
    else:
      label = None
    ax.plot([edge,edges[e+1]], [bins[e],bins[e+1]], '.-', color=color, label=label, alpha=alpha)

def plotPDF(edges, bins, ax, label, color, s='.', alpha=1.0):
  # like a pdf, with error bars for bin width
  mids = 0.5*(edges[1:]+edges[:-1])
  lows = edges[:-1]
  highs = edges[1:]
  ax.errorbar( mids, bins, xerr=[mids-lows, highs-mids], fmt=s+'-', color=color, label=label, alpha=alpha)

# Enabling plotting will help visualize the signals that are tested
#    in the event they fail tests. Plotting should not be enabled in
#    the regression system as this point.
plotting = False
if plotting:
  import matplotlib.pyplot as plt
  fig, (ax,ax2) = plt.subplots(1, 2, figsize=(16,12))

N = int(1e4)
# NOTE: evaluating is slow for 1e5, and very slow at 1e6

# Beta distribution of data, skewed low
dist = stats.lognorm(0.3)
#dist = stats.beta(2.0, 3.0)
if plotting:
  x = np.linspace(dist.ppf(0.001),dist.ppf(0.999),N)
  pdf = dist.pdf(x)
  cdf = dist.cdf(x)
  ax.plot(x,cdf,'k-',label='true beta', lw=3)
  ax2.plot(x,pdf,'k-',label='true beta', lw=3)

# random samples
data=pd.read_csv("signal.csv")
data=data.e_demand.values

if plotting:
  opdf, ocdf = makeCDF(data)
  plotCDF(ocdf[0], ocdf[1], ax, 'data', 'C0')
  plotPDF(opdf[0], opdf[1], ax2, 'data', 'C0', s='x')

# characterize
params = arma._trainCDF(data)
if plotting:
  ebins = params['bins']
  ecdf = params['cdf']
  epdf = params['pdf']
  plotCDF(ebins, ecdf, ax, 'empirical', 'C1')
  plotPDF(ebins, epdf, ax2, 'empirical', 'C1')

# gaussian for reference
if plotting:
  gauss = stats.norm()
  gx = np.linspace(-3,3,N)
  gpdf = gauss.pdf(gx)
  gcdf = gauss.cdf(gx)
  ax.plot(gx,gcdf,'k:',label='true normal', lw=3)
  ax2.plot(gx,gpdf,'k:',label='true normal', lw=3)

# gaussianize it
normed = arma._normalizeThroughCDF(data,params)
if plotting:
  npdf, ncdf = makeCDF(normed)
  plotCDF(ncdf[0], ncdf[1], ax, 'normed', 'C2')
  plotPDF(npdf[0], npdf[1], ax2, 'normed', 'C2')

# undo gaussian
denormed = arma._denormalizeThroughCDF(normed, params)
if plotting:
  dpdf, dcdf = makeCDF(denormed)
  plotCDF(dcdf[0], dcdf[1], ax, 'denormed', 'C3')
  plotPDF(dpdf[0], dpdf[1], ax2, 'denormed', 'C3')

# pre-normalized and post-normalized should be the same
delta = np.abs(data - denormed)
checkArray('CDF/ICDF consistency',data,denormed,float,tol=1e-12,update=True)

if plotting:
  ax.legend(loc=0)
  ax2.legend(loc=0)
  ax.set_title('CDF')
  ax2.set_title('PDF')
  plt.show()

# train ARMA on data and check CDFs of results
rom, arma = createARMA(['a'], 't', 0, 0, [])
featureVals = np.zeros(1)
targetVals = np.zeros([1,len(data),2])
# "a"
targetVals[0,:,0] = data
# "t"
t = np.arange(len(data))
targetVals[0,:,1] = t
arma.__trainLocal__(featureVals,targetVals)
nsamp = 10
samples = np.zeros([nsamp,len(data)])
for n in range(nsamp):
  ev = arma.__evaluateLocal__(np.array([1.0]))
  samples[n,:] = ev['a']
# Enabling plotting will help visualize the signals that are tested
#    in the event they fail tests. Plotting should not be enabled in
#    the regression system as this point.
plotting = False
if plotting:
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()
  figC, (axC1,axC2) = plt.subplots(1,2)
  # samples
  ax.plot(t, data, 'k-', label='original')
ostats = (np.average(data), np.std(data))
for n in range(nsamp):
  stats = (np.average(samples[n,:]), np.std(samples[n,:]))
  checkFloat('Mean, sample {}'.format(n), ostats[0], stats[0], tol=3e-1)
  checkFloat('Std, sample {}'.format(n), ostats[1], stats[1], tol=6e-1)
  if plotting:
    ax.plot(t, samples[n,:], '-', color='C1', label='sample', alpha=0.2)
    pdf,cdf = makeCDF(samples[n,:])
    # PDF/CDF
    plotPDF(pdf[0], pdf[1], axC2, 'sample'+str(n), 'C1', alpha=0.3)
    plotCDF(cdf[0], cdf[1], axC1, 'sample'+str(n), 'C1', alpha=0.3)
if plotting:
  # original
  pdf,cdf = makeCDF(data)
  plotPDF(pdf[0], pdf[1], axC2, 'original', 'k')
  plotCDF(cdf[0], cdf[1], axC1, 'original', 'k')
  ax.legend(loc=0)
  plt.show()


#############################################
#            RESEEDCOPIES, ENGINE           #
#############################################

testVal = arma._trainARMA(data)
arma.amITrained = True
signal1 = arma._generateARMASignal(testVal)
signal2 = arma._generateARMASignal(testVal)

#Test the reseed = False

armaReF = arma
armaReF.reseedCopies = False
pklReF = pk.dumps(armaReF)
unpkReF = pk.loads(pklReF)
#signal 3 and 4 should be the same
signal3 = armaReF._generateARMASignal(testVal)
signal4 = unpkReF._generateARMASignal(testVal)
for n in range(len(data)):
  checkFloat('signal 3, signal 4 ind{}'.format(n), signal3[n], signal4[n], tol=1e-5)

#Test the reseed = True

arma.reseedCopies = True
pklReT = pk.dumps(arma)
unpkReT = pk.loads(pklReT)
#signal 5 and 6 should not be the same
signal5 = arma._generateARMASignal(testVal)
signal6 = unpkReT._generateARMASignal(testVal)
for n in range(len(data)):
  checkTrue('signal 5, signal 6 ind{}'.format(n),signal5[n]!=signal6[n])

# Test the engine with seed

eng = randomUtils.newRNG()
arma.setEngine(eng, seed=901017)
signal7 = arma._generateARMASignal(testVal)

sig7 = [0.39975177, -0.14531468,  0.13138866, -0.56565224,  0.06020252,
        0.60752306, -0.29076173, -1.1758456,   0.41108591, -0.05735384]
for n in range(10):
  checkFloat('signal 7, evaluation ind{}'.format(n), signal7[n], sig7[n], tol=1e-7)

#################
# TODO UNTESTED #
#################
# - Segmented
# - VARMA construction
# - Analytic VARMA/ARMA variances
# - Fourier analytic coefficients
# - Signal Reconstruction


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
