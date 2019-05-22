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
'''
 This file contains the random number generating methods used in the framework.
 created on 07/15/2017
 @author: talbpaul
'''

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import numpy as np
from collections import deque, defaultdict

from utils.utils import findCrowModule

# in general, we will use Crow for now, but let's make it easy to switch just in case it is helpfull eventually.
# Numoy stochastic environment can not pass the test as this point
stochasticEnv = 'crow'
#stochasticEnv = 'numpy'

class BoxMullerGenerator:
  """
    Iterator class for the Box-Muller transform
  """
  def __init__(self):
    """
      Constructor.
      @ In, engine, instance, optional, random number generator
      @ Out, None
    """
    self.queue = defaultdict(deque)

  def generate(self,engine=None):
    """
      Yields a normally-distributed pseudorandom value
      @ In, engine, instance, optional, random number generator
      @ Out, generate, float, random value
    """
    if len(self.queue[engine]) == 0:
      #calculate new values
      self.queue[engine].extend(self.createSamples(engine=engine))
    return self.queue[engine].pop() #no need to pop left, as they're independent and all get used

  def createSamples(self,engine=None):
    """
      Sample calculator.  Because Box Muller does batches of 2, add them to a queue.
      @ In, engine, instance, optional, random number generator.
      @ Out, (z1,z2), tuple, two independent random values
    """
    u1,u2 = random(2,engine=engine)
    z1 = np.sqrt(-2.*np.log(u1))*np.cos(2.*np.pi*u2)
    z2 = np.sqrt(-2.*np.log(u1))*np.sin(2.*np.pi*u2)
    return z1,z2

  def testSampling(self, n=1e5,engine=None):
    """
      Tests distribution of samples over a large number.
      @ In, n, int, optional, number of samples to test with
      @ In, engine, instance, optional, random number generator
      @ Out, mean, float, mean of sample set
      @ Out, stdev, float, standard deviation of sample set
    """
    n = int(n)
    samples = np.array([self.generate(engine=engine) for _ in range(n)])
    mean = np.average(samples)
    stdev = np.std(samples)
    return mean,stdev

if stochasticEnv == 'numpy':
  npStochEnv = np.random.RandomState()
else:
  crowStochEnv = findCrowModule('randomENG').RandomClass()
  # this is needed for now since we need to split the stoch enviroments
  distStochEnv = findCrowModule('distribution1D').DistributionContainer.instance()
  boxMullerGen = BoxMullerGenerator()

def randomSeed(value,seedBoth=False,engine=None):
  """
    Function to get a random seed
    @ In, value, float, the seed
    @ In, engine, instance, optional, random number generator
    @ In, seedBoth, bool, optional, if True then seed both random environments
    @ Out, None
  """
  # we need a flag to tell us  if the global numpy stochastic environment is needed to be changed
  replaceGlobalEnv=False
  ## choose an engine if it is none
  if engine is None:
    if stochasticEnv == 'crow':
      distStochEnv.seedRandom(value)
      engine=crowStochEnv
    elif stochasticEnv == 'numpy':
      replaceGlobalEnv=True
      global npStochEnv
      # global npStochEvn is needed in numpy environment here
      # to prevent referenced before assignment in local loop
      engine = npStochEnv

  if isinstance(engine, np.random.RandomState):
    engine = np.random.RandomState(value)
  elif isinstance(engine, findCrowModule('randomENG').RandomClass):
    engine.seed(value)
    if seedBoth:
      np.random.seed(value+1) # +1 just to prevent identical seed sets
  if stochasticEnv== 'numpy' and replaceGlobalEnv:
    npStochEnv= engine
  if replaceGlobalEnv:
    print('randomUtils: Global random number seed has been changed to',value)

def random(dim=1,samples=1,keepMatrix=False,engine=None):
  """
    Function to get a single random value, an array of random values, or a matrix of random values, on [0,1]
    @ In, dim, int, optional, dimensionality of samples
    @ In, samples, int, optional, number of arrays to deliver
    @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
    @ In, engine, instance, optional, random number generator
    @ Out, vals, float, random normal number (or np.array with size [n] if n>1, or np.array with size [n,samples] if sampels>1)
  """
  engine=getEngine(engine)
  dim = int(dim)
  samples = int(samples)
  if isinstance(engine, np.random.RandomState):
    vals = engine.rand(samples,dim)
  elif isinstance(engine, findCrowModule('randomENG').RandomClass):
    vals = np.zeros([samples,dim])
    for i in range(len(vals)):
      for j in range(len(vals[0])):
        vals[i][j] = engine.random()
  # regardless of stoch env
  if keepMatrix:
    return vals
  else:
    return _reduceRedundantListing(vals,dim,samples)

def randomNormal(dim=1,samples=1,keepMatrix=False,engine=None):
  """
    Function to get a single random value, an array of random values, or a matrix of random values, normally distributed
    @ In, dim, int, optional, dimensionality of samples
    @ In, samples, int, optional, number of arrays to deliver
    @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
    @ In, engine, instance, optional, random number generator
    @ Out, vals, float, random normal number (or np.array with size [n] if n>1, or np.array with size [n,samples] if sampels>1)
  """
  engine=getEngine(engine)
  dim = int(dim)
  samples = int(samples)
  if isinstance(engine, np.random.RandomState):
    vals = engine.randn(samples,dim)
  elif isinstance(engine, findCrowModule('randomENG').RandomClass):
    vals = np.zeros([samples,dim])
    for i in range(len(vals)):
      for j in range(len(vals[0])):
        vals[i,j] = boxMullerGen.generate(engine=engine)
  if keepMatrix:
    return vals
  else:
    return _reduceRedundantListing(vals,dim,samples)

def randomIntegers(low,high,caller,engine=None):
  """
    Function to get a random integer
    @ In, low, int, low boundary
    @ In, high, int, upper boundary
    @ In, caller, instance, object requesting the random integers
    @ In, engine, instance, optional, random number generator
    @ Out, rawInt, int, random int
  """
  engine=getEngine(engine)
  if isinstance(engine, np.random.RandomState):
    return engine.randint(low,high=high+1)
  elif isinstance(engine, findCrowModule('randomENG').RandomClass):
    intRange = high-low
    rawNum = low + random(engine=engine)*intRange
    rawInt = int(round(rawNum))
    if rawInt < low or rawInt > high:
      caller.raiseAMessage("Random int out of range")
      rawInt = max(low,min(rawInt,high))
    return rawInt
  else:
    raise TypeError('Engine type not recognized! {}'.format(type(engine)))

def randomPermutation(l,caller,engine=None):
  """
    Function to get a random permutation
    @ In, l, list, list to be permuted
    @ In, caller, instance, the caller
    @ In, engine, instance, optional, random number generator
    @ Out, newList, list, randomly permuted list
  """
  engine=getEngine(engine)
  if isinstance(engine, np.random.RandomState):
    return engine.permutation(l)
  elif isinstance(engine, findCrowModule('randomENG').RandomClass):
    newList = []
    oldList = l[:]
    while len(oldList) > 0:
      newList.append(oldList.pop(randomIntegers(0,len(oldList)-1,caller,engine=engine)))
    return newList

def randPointsOnHypersphere(dim,samples=1,r=1,keepMatrix=False,engine=None):
  """
    obtains random points on the surface of a hypersphere of dimension "n" with radius "r".
    see http://www.sciencedirect.com/science/article/pii/S0047259X10001211
    "On decompositional algorithms for uniform sampling from n-spheres and n-balls", Harman and Lacko, 2010, J. Multivariate Analysis
    @ In, dim, int, the dimensionality of the hypersphere
    @ In, samples, int, optional, the number of samples desired
    @ In, r, float, optional, the radius of the hypersphere
    @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
    @ In, engine, instance, optional, random number generator
    @ Out, pts, np.array(np.array(float)), random points on the surface of the hypersphere [sample#][pt]
  """
  engine=getEngine(engine)
  ## first fill random samples
  pts = randomNormal(dim,samples=samples,keepMatrix=True,engine=engine)
  ## extend radius, place inside sphere through normalization
  rnorm = float(r)/np.linalg.norm(pts,axis=1)
  pts *= rnorm[:,np.newaxis]
  #TODO if all values in any given sample are 0,
  #       this produces an unphysical result, so we should resample;
  #       however, this probability is miniscule and the speed benefits of skipping checking loop seems worth it.
  if keepMatrix:
    return pts
  else:
    return _reduceRedundantListing(pts,dim,samples)
  return pts

def randPointsInHypersphere(dim,samples=1,r=1,keepMatrix=False,engine=None):
  """
    obtains a random point internal to a hypersphere of dimension "n" with radius "r"
    see http://www.sciencedirect.com/science/article/pii/S0047259X10001211
    "On decompositional algorithms for uniform sampling from n-spheres and n-balls", Harman and Lacko, 2010, J. Multivariate Analysis
    @ In, dim, int, the dimensionality of the hypersphere
    @ In, r, float, the radius of the hypersphere
    @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
    @ In, engine, instance, optional, random number generator
    @ Out, pt, np.array(float), a random point on the surface of the hypersphere
  """
  engine=getEngine(engine)
  #sample on surface of n+2-sphere and discard the last two dimensions
  pts = randPointsOnHypersphere(dim+2,samples=samples,r=r,keepMatrix=True,engine=engine)[:,:-2]
  if keepMatrix:
    return pts
  else:
    return _reduceRedundantListing(pts,dim,samples)
  return pts

def newRNG(env=None):
  """
    Provides a new instance of the random number generator.
    @ In, env, string, optional, type of random number generator.  Defaults to global option stored in "stochasticEnv".
    @ Out, engine, object, RNG producer
  """
  if env is None:
    env = stochasticEnv
  if env == 'crow':
    engine = findCrowModule('randomENG').RandomClass()
  elif env == 'numpy':
    engine = np.random.RandomState()
  return engine

### internal utilities ###

def _reduceRedundantListing(data,dim,samples):
  """
    Adjusts data to be intuitive for developers.
     - if dim = samples = 1: returns a float
     - if dim > 1 but samples = 1: returns a 1D numpy array of floats
     - otherwise: returns a 2D numpy array indexed by [sample][dim]
    @ In, data, numpy.array, two-dimensional array indexed by [sample][dim]
    @ In, dim, int, dimensionality of each sample
    @ In, samples, int, number of samples taken
    @ Out, data, np.array, shape and size described above in method description.
  """
  if dim==1 and samples==1: #user expects single float
    return data[0][0]
  elif samples==1: #user expects array of floats
    return data[0]
  #elif dim==1: #potentially user expects array of floats, but probably wants array of single-entry arrays
  #  return data[:,0]
  else:
    return data

def getEngine(eng):
  """
   Choose an engine if it is none and raise error if engine type not recognized
   @ In, engine, instance, random number generator
   @ Out, engine, instance, random number generator
  """
  if eng is None:
    if stochasticEnv == 'numpy':
      eng = npStochEnv
    elif stochasticEnv == 'crow':
      eng = crowStochEnv
  if not isinstance(eng, np.random.RandomState) and not isinstance(eng, findCrowModule('randomENG').RandomClass):
    raise TypeError('Engine type not recognized! {}'.format(type(eng)))
  return eng
