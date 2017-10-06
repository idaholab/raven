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

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os,sys

import numpy as np
import xarray as XR

frameworkDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),os.pardir,os.pardir,os.pardir,'framework'))
sys.path.append(frameworkDir)

from utils import CachedXArray

print (CachedXArray)
#-------------------
# utility functions
#-------------------

def model(inputs):
  """
    Simple model to produce demonstration data for unit testins.
    @ In, inputs, list, list of float values
    @ Out, scalar, float, single value
    @ Out, pivot, list, list of floats corresponding to a coordinate variable like "time"
    @ Out, vector, list, list of floats with values associated with each coordinate value
  """
  scalar = np.sum(inputs)
  pivot = range(10)
  vector = list(p*scalar for p in pivot)
  return scalar,pivot,vector

runIndex = 0 #global counter for number of realizations

def runModel(seed=2,random=False):
  global runIndex
  if random:
    inputs = np.random.rand(3)
  else:
    inputs = range(seed,seed+3)
  s,p,v = model(inputs)
  results = {}
  results['scalar']   = s #XR.DataArray([s],dims=['sample']       ,coords={'sample':[runIndex]         })
  #results['vector'] = XR.DataArray([v],dims=['sample','time'],coords={'sample':[runIndex],'time':p})
  #realization = XR.Dataset(data_vars = results)
  runIndex += 1
  return results #alization

def checkSame(test,gold,msg):
  if test == gold:
    passed['pass']+=1
  else:
    print(msg,'| Expected',gold,'but got',test)
    passed['fail']+=1

def checkSameVector(test,gold,msg):
  ok = False
  if len(test) == len(gold):
    if all(test[i] == gold[i] for i in range(len(test))):
      ok = True
  if ok:
    passed['pass']+=1
  else:
    print(msg,'| Expected',gold,'but got',test)
    passed['fail']+=1

#------------
# unit tests
#------------

passed = {'pass':0, 'fail':0}

# construction
# you can't contruct the object without initial data
#firstRealization = runModel()
cxa = CachedXArray.CachedDataset(cacheSize=10,prealloc=True,entries=['scalar'])
l = len(cxa._cache)
checkSame(l,10,'Setting cache size')

# set the cache size
cxa.setCacheSize(10)
l = len(cxa._cache)
checkSame(l,10,'Setting cache size')

# add one entry
rlz = runModel()
print('realization:',rlz)
print('cache:')
print(cxa._cache)
print('')
cxa.append(rlz)
print('cache, appended:')
print(cxa._cache)
#check data
#dat = cxa._cache[0]
#checkSame(dat.sample.values,[0],'Cached entry "sample" values')
#checkSame(dat.scalar.values,[9],'Cached entry "scalar" values')
#checkSameVector(dat.vector[0].values,[0,9,18,27,36,45,54,63,72,81],'Cached entry "vector" values')
#checkSameVector(dat.time.values,range(10),'Cached entry "vector" values')
#check the entry at sample, time
#checkSame(dat.sel(sample=0,time=5).vector,45,'Cached entry one value')

# add a few entries
print('')
print('cache, added:')
for i in range(5):
  rlz = runModel(seed=i)
  cxa.append(rlz)
print(cxa._cache)
#check a few data points
#checkSame(cxa._cache[1].sel(sample=1,time=5).vector,15,'Cached entry values 1')
#checkSame(cxa._cache[3].sel(sample=3,time=9).vector,81,'Cached entry values 2')
#checkSame(cxa._cache[5].sel(sample=5,time=3).vector,45,'Cached entry values 3')
#checkSame(cxa._cache[6],None,'Cached entry values 4')

print('')
print('main, added to flush:')
# add entries until flushed, twice
for i in range(6,21):
  rlz = runModel(seed=i)
  cxa.append(rlz)

print(cxa)
print('')
print('cache')
print(cxa._cache)
import sys; sys.exit()

# check length
l = len(cxa)
checkSame(len(cxa),21,'Length of array (number of samples)')
# get dataset

#check a few points in the main data object
#  NOTE that these values match those above!
dat = cxa.asDataset()
checkSame(dat.sel(sample=1,time=5).vector,15,'Cached entry values 1')
checkSame(dat.sel(sample=3,time=9).vector,81,'Cached entry values 2')
checkSame(dat.sel(sample=5,time=3).vector,45,'Cached entry values 3')

print('Results:',passed)
sys.exit(passed['fail'])
