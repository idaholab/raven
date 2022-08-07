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
 This file contains the random number generating methods used in the framework.
 created on 07/15/2017
 @author: talbpaul
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import math
import threading
from collections import deque, defaultdict
import numpy as np
import copy

from .utils import findCrowModule
from . import mathUtils
from ..CustomDrivers.DriverUtils import setupCpp

# in general, we will use Crow for now, but let's make it easy to switch just in case it is helpful eventually.
# Numpy stochastic environment can not pass the test as this point
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
    self.__queueLock = threading.RLock()

  def generate(self,engine=None):
    """
      Yields a normally-distributed pseudorandom value
      @ In, engine, instance, optional, random number generator
      @ Out, generate, float, random value
    """
    with self.__queueLock:
      if len(self.queue[engine]) == 0:
        #calculate new values
        self.queue[engine].extend(self.createSamples(engine=engine))
      val = self.queue[engine].pop()
    return val

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

class CrowRNG:
  """ Wraps crow RandomClass to make it serializable """
  def __init__(self, engine=None):
    """
      Constructor
      @ In, engine, RandomClass, optional, will wrap the given engine if provided, otherwise a new engine is created
      @ Out, None
    """
    if engine is None:
      self._engine = findCrowModule('randomENG').RandomClass()
    elif isinstance(engine, findCrowModule('randomENG').RandomClass):
      self._engine = engine
    else:
      raise TypeError(f'Object of unknown type {type(engine)} cannot be wrapped by CrowRNG class!')
    self._seed = abs(int(self._engine.get_rng_seed()))

  def __getstate__(self):
    """
      Get state for serialization
      @ In, None
      @ Out, d, dict, object instance state
    """
    d = copy.copy(self.__dict__)
    eng = d.pop('_engine')  # remove RNG engine from class instance
    d['count'] = eng.get_rng_state()
    return d

  def __setstate__(self, d):
    """
      Set object instance state
      @ In, d, dict, object state
      @ Out, None
    """
    count = d.pop('count')
    self.__dict__.update(d)
    self._engine = findCrowModule('randomENG').RandomClass()  # reinstantiate RNG engine
    self._engine.seed(self._seed)
    self._engine.forward_seed(count)

  def seed(self, value):
    """
      Wrapper for RandomClass.seed()
      @ In, value, int, RNG seed
      @ Out, None
    """
    self._seed = abs(int(value))
    self._engine.seed(self._seed)  # takes unsigned long

  def random(self):
    """
      Wrapper for RandomClass.random()
      @ In, None
      @ Out, float, random number from RNG engine
    """
    return self._engine.random()  # returns double

  def getRNGState(self):
    """
      Wrapper for RandomClass.get_rng_state()
      @ In, None
      @ Out, int, RNG state
    """
    return self._engine.get_rng_state()  # returns int

  def forwardSeed(self, counts):
    """
      Wrapper for RandomClass.forward_seed()
      @ In, counts, int, number of random states to progress
      @ Out, None
    """
    self._engine.forward_seed(counts)  # takes unsigned int

  def getRNGSeed(self):
    """
      Wrapper for RandomClass.get_rng_seed()
      @ In, None
      @ Out, int, RNG seed value
    """
    val = self._engine.get_rng_seed()  # returns int
    self._seed = abs(int(val))
    return self._seed

if stochasticEnv == 'numpy':
  npStochEnv = np.random.RandomState()
else:
  setupCpp()
  crowStochEnv = CrowRNG()
  # this is needed for now since we need to split the stoch environments
  distStochEnv = findCrowModule('distribution1D').DistributionContainer.instance()
  boxMullerGen = BoxMullerGenerator()

#
# Utilities
#
#
def randomSeed(value, seedBoth=False, engine=None):
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
      engine = crowStochEnv
    elif stochasticEnv == 'numpy':
      replaceGlobalEnv = True
      global npStochEnv
      # global npStochEvn is needed in numpy environment here
      # to prevent referenced before assignment in local loop
      engine = npStochEnv

  if isinstance(engine, np.random.RandomState):
    engine = np.random.RandomState(value)
  elif isinstance(engine, CrowRNG):
    engine.seed(value)
    if seedBoth:
      np.random.seed(value+1) # +1 just to prevent identical seed sets
  if stochasticEnv == 'numpy' and replaceGlobalEnv:
    npStochEnv = engine
  if replaceGlobalEnv:
    print('randomUtils: Global random number seed has been changed to',value)

def forwardSeed(count, engine):
  """
    Function to advance the state of a random number generator engine
    @ In, count, int, number of steps to advance the RNG
    @ In, engine, np.random.RandomState or CrowRNG, RNG engine to modify
    @ Out, None
  """
  if isinstance(engine, np.random.RandomState):
    engine.rand(count)
  elif isinstance(engine, CrowRNG):
    engine.forwardSeed(count)

def random(dim=1, samples=1, keepMatrix=False, engine=None):
  """
    Function to get a single random value, an array of random values, or a matrix of random values, on [0,1]
    @ In, dim, int, optional, dimensionality of samples
    @ In, samples, int, optional, number of arrays to deliver
    @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
    @ In, engine, instance, optional, random number generator
    @ Out, vals, float, random normal number (or np.array with size [n] if n>1, or np.array with size [n,samples] if sampels>1)
  """
  engine = getEngine(engine)
  dim = int(dim)
  samples = int(samples)
  if isinstance(engine, np.random.RandomState):
    vals = engine.rand(samples,dim)
  elif isinstance(engine, CrowRNG):
    vals = np.zeros([samples, dim])
    for i in range(len(vals)):
      for j in range(len(vals[0])):
        vals[i][j] = engine.random()
  # regardless of stoch env
  if keepMatrix:
    return vals
  else:
    return _reduceRedundantListing(vals, (samples, dim))

def randomNormal(size=(1,), keepMatrix=False, engine=None):
  """
    Function to get a single random value, an array of random values, or a matrix of random values, normally distributed
    @ In, size, int or tuple, optional, shape of the samples to return
      (if int, an array of samples will be returned if size>1, otherwise a float if keepMatrix is false)
    @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
    @ In, engine, instance, optional, random number generator
    @ Out, vals, float, random normal number (or np.array with size [n] if n>1, or np.array with size [n,samples] if sampels>1)
  """
  engine = getEngine(engine)
  if isinstance(size, int):
    size = (size, )
  if isinstance(engine, np.random.RandomState):
    vals = engine.randn(*size)
  elif isinstance(engine, CrowRNG):
    vals = np.zeros(np.prod(size))
    for i in range(len(vals)):
      vals[i] = boxMullerGen.generate(engine=engine)
    vals.shape = size
  if keepMatrix:
    return vals
  else:
    return _reduceRedundantListing(vals,size)

def randomMultivariateNormal(cov, size=1, mean=None):
  """
    Provides a random sample from a multivariate distribution.
    @ In, cov, np.array, covariance matrix (must be square, positive definite)
    @ In, size, int, optional, number of samples to return
    @ In, mean, np.array, means for distributions (must be length of 1 side of covar matrix == len(cov[0]))
    @ Out, vals, np.array, array of samples with size [n_samples, len(cov[0])]
  """
  dims = cov.shape[0]
  if mean is None:
    mean = np.zeros(dims)
  eps = 10 * sys.float_info.epsilon
  covEps = cov + eps * np.identity(dims)
  decomp = np.linalg.cholesky(covEps)
  randSamples = randomNormal(size=(dims, size)).reshape((dims, size))
  vals = mean + np.dot(decomp, randSamples)
  return vals

def randomIntegers(low, high, caller=None, engine=None):
  """
    Function to get a random integer
    @ In, low, int, low boundary
    @ In, high, int, upper boundary
    @ In, caller, instance, optional, object requesting the random integers
    @ In, engine, instance, optional, optional, random number generator
    @ Out, rawInt, int, random int
  """
  engine = getEngine(engine)
  if isinstance(engine, np.random.RandomState):
    return engine.randint(low, high=high+1)
  elif isinstance(engine, CrowRNG):
    intRange = high - low + 1.0
    rawNum = low + random(engine=engine)*intRange
    rawInt = math.floor(rawNum)
    if rawInt < low or rawInt > high:
      if caller:
        caller.raiseAMessage("Random int out of range")
      rawInt = max(low, min(rawInt, high))
    return rawInt
  else:
    raise TypeError('Engine type not recognized! {}'.format(type(engine)))

def randomChoice(array, size = 1, replace = True, engine = None):
  """
    Generates a random sample or a sequence of random samples from a given array-like (list or such) or N-D array
    This equivalent to np.random.choice but extending the functionality to N-D arrays
    @ In, array, list or np.ndarray, the array from which to pick
    @ In, size, int, optional, the number of samples to return
    @ In, replace, bool, optional, allows replacement if True, default is True
    @ In, engine, instance, optional, optional, random number generator
    @ Out, selected, object, the random choice (1 element) or a list of elements
  """
  assert(hasattr(array,"shape") or isinstance(array,list))

  if not replace:
    if hasattr(array,"shape"):  # TODO: not a problem actually. Should be able to use numpy.random.RandomState.choice(a, replace=False)
      raise RuntimeError("Option with replace False not available for ndarrays")
    if len(array) < size:
      raise RuntimeError("array size < of number of requested samples (size)")

  sel = []
  coords = array
  for _ in range(size):
    if hasattr(array,"shape"):
      coord = tuple([randomIntegers(0, dim-1, engine=engine) for dim in coords.shape])
      sel.append(coords[coord])
    else:
      sel.append(coords[randomIntegers(0, len(coords)-1, engine=engine)])
    if not replace:
      coords.remove(sel[-1])
  selected = sel[0] if size == 1 else sel
  return selected

def randomPermutation(l,caller,engine=None):
  """
    Function to get a random permutation
    @ In, l, list, list to be permuted
    @ In, caller, instance, the caller
    @ In, engine, instance, optional, random number generator
    @ Out, newList, list, randomly permuted list
  """
  engine = getEngine(engine)
  if isinstance(engine, np.random.RandomState):
    return engine.permutation(l)
  elif isinstance(engine, CrowRNG):
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
  pts = randomNormal(size=(samples, dim),keepMatrix=True,engine=engine)
  ## extend radius, place inside sphere through normalization
  rnorm = float(r)/np.linalg.norm(pts,axis=1)
  pts *= rnorm[:,np.newaxis]
  #TODO if all values in any given sample are 0,
  #       this produces an unphysical result, so we should resample;
  #       however, this probability is miniscule and the speed benefits of skipping checking loop seems worth it.
  if keepMatrix:
    return pts
  else:
    return _reduceRedundantListing(pts,(samples, dim))
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
    return _reduceRedundantListing(pts,(samples, dim))
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
    engine = CrowRNG()
  elif env == 'numpy':
    engine = np.random.RandomState()
  return engine

### internal utilities ###

def _reduceRedundantListing(data,size):
  """
    Adjusts data to be intuitive for developers.
     - if np.prod(size) => dim = samples = 1: returns a float
     - if size[1,...,n] > 1 but size[0] (samples) = 1: returns a 1D numpy array of floats
     - otherwise: returns a  numpy array indexed by the original shape
    @ In, data, numpy.array, n-dimensional array indexed by [sample, :, ...,n]
    @ In, dim, int, dimensionality of each sample
    @ In, samples, int, number of samples taken
    @ Out, data, np.array, shape and size described above in method description.
  """
  if np.prod(size) == 1: #user expects single float
    return data.flatten()[0]
  elif size[0]==1: #user expects array of floats (or matrix)
    return data[0]
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
  if not isinstance(eng, np.random.RandomState) and not isinstance(eng, CrowRNG):
    raise TypeError('Engine type not recognized! {}'.format(type(eng)))
  return eng

def randomPerpendicularVector(vector):
  """
    Finds a random vector perpendicular to the given vector
    Uses definition of dot product orthogonality:
    0 = sum_i (p_i * g_i)
    p_i = rand() forall i != n
    p_n = -1/g_n * sum_i(p_i * g_i) forall i != n
    @ In, vector, np.array, ND vector
    @ Out, perp, np.array, perpendicular vector
  """
  # sanity check
  numNonZero = np.count_nonzero(vector)
  if not numNonZero:
    raise RuntimeError('Provided vector is the zero vector!')
  N = len(vector)
  indices = np.arange(N)
  nonZeroMap = vector != 0
  # choose a random NONZERO index to be dependent (don't divide by zero, mate)
  depIndex = indices[nonZeroMap][randomIntegers(0, numNonZero - 1, None)]
  # random values for all but chosen variable
  perp = randomNormal(N)
  # cheat some math, zero out the random index term by setting the perp value to 0
  perp[depIndex] = 0
  dotProd = np.dot(vector, perp)
  perp[depIndex] = - dotProd / vector[depIndex]
  return perp
