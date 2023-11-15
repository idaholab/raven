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
from ..CustomDrivers.DriverUtils import setupCpp

# As of 2023-11-13, the crow and numpy random environments produce identical outputs for the same seed.
# Numpy version 1.22.4 is used. The numpy is environment is now the default, and crow will be removed
# in the future.
# stochasticEnv = 'crow'
stochasticEnv = 'numpy'

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

  def generate(self, size, engine=None):
    """
      Yields a normally-distributed pseudorandom value
      @ In, size, tuple, number of values to generate
      @ In, engine, instance, optional, random number generator
      @ Out, values, np.ndarray, random value
    """
    values = np.zeros(size)
    with self.__queueLock:
      queueLength = len(self.queue[engine])
      if queueLength < size:
        #calculate new values
        # We want to add only as many new values as we need to the queue, but Box Muller does them in pairs.
        # Asking for ceil((size - len(queue)) / 2) evaluations of Box Muller will give us enough values to
        # satisfy the request.
        genSize = np.ceil((size - queueLength) / 2).astype(int)
        # Using extendleft instead of extend so that the values are added behind the existing values in the deque.
        self.queue[engine].extendleft(self.createSamples(size=genSize, engine=engine))
    values = np.array([self.queue[engine].pop() for _ in range(size)])
    return values

  def createSamples(self, size, engine=None):
    """
      Sample calculator.  Because Box Muller does batches of 2, 2*size numbers are generated. If size
      is 1, a non-vectorized version is used. Otherwise, a vectorized implementation calculates all
      of the random numbers at once. The vectorized implementation is much faster for large values
      of size but adds overhead for small values. When size is greater than 1, the returned values
      are interleaved to match the order of repeated calls to this function with size=1.
      @ In, size, int, number of samples to create
      @ In, engine, instance, optional, random number generator.
      @ Out, z, tuple or np.ndarray, two independent random values
    """
    # This is a little ugly, but the original code (if block) is the faster way to handle the size=1
    # case (~25% faster). However, the vectorized code (else block) is up to 100x faster if generating
    # many samples at once. Implementing like this allows us to get the best of both worlds.
    if size == 1:
      u1,u2 = random(2,engine=engine)
      z1 = np.sqrt(-2.*np.log(u1))*np.cos(2.*np.pi*u2)
      z2 = np.sqrt(-2.*np.log(u1))*np.sin(2.*np.pi*u2)
      z = (z2, z1)  # see note below for why z1 and z2 are reversed
    else:
      u = random(2, size, engine=engine)
      r = np.sqrt(-2. * np.log(u[:, 0]))
      theta = 2. * np.pi * u[:, 1]
      z1 = r * np.cos(theta)
      z2 = r * np.sin(theta)
      # The original code was returning z1,z2, so the z1 and z2 produced here need to be combined
      # into a single 1-d array where the values of z1 and z2 are interleaved. Also, because of the
      # first in-last out nature of the deque, we need to reverse the order of the values so that
      # they come out in the expected order.
      z = np.vstack((z2, z1)).T.flatten()
    return z

  def testSampling(self, n=1e5,engine=None):
    """
      Tests distribution of samples over a large number.
      @ In, n, int, optional, number of samples to test with
      @ In, engine, instance, optional, random number generator
      @ Out, mean, float, mean of sample set
      @ Out, stdev, float, standard deviation of sample set
    """
    n = int(n)
    samples = self.generate(n, engine=engine)
    mean = np.average(samples)
    stdev = np.std(samples)
    return mean,stdev

class CrowRNG:
  """ Wraps crow RandomClass to make it serializable """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    self._engine = findCrowModule('randomENG').RandomClass()
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

  def random(self, size):
    """
      Wrapper for RandomClass.random()
      @ In, size, tuple, size of array to return
      @ Out, vals, np.ndarray, random numbers from RNG engine
    """
    vals = np.zeros(size)
    for i in range(size[0]):
      for j in range(size[1]):
        vals[i][j] = self._engine.random()
    return vals

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

# FIXME: A wrapper class for numpy.random.Generator is used here instead of using the Generator directly
# in order to reproduce the same outputs as CrowRNG.random(). CrowRNG, which uses the 32-bit
# boost::random::mt19937 under the hood in Crow, produces 64-bit floats by dividing a 32-bit integer
# by the maximum value of an unsigned 32-bit integer; this division produces a 64-bit float. This is
# far from ideal, since this does not actually contain 64 bits of randomness. However, fixing this
# or using an RNG which does not produce identical outputs to CrowRNG.random() would require enormous
# effort to fix all of the tests which rely on the current behavior. At the time of writing, that is
# approximately 400 of about 900 tests! If anybody has the availability to fix that many tests, please
# do it! Switching the util functions in this file to directly use the methods of np.random.Generator
# really streamline things and probably provide a reasonable speed boost. But for now, we'll go with
# the wrapper option so we don't have to regold all those tests.
#   j-bryan, 2023-11-13
class NumpyRNG:
  """ Wraps numpy.random.Generator to provide an interface similar to CrowRNG """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    self._engine = None
    self._seed = None
    self.seed(5489)  # default seed of boost::random::mt19937

  def seed(self, value):
    """
      Reseeds the RNG
      @ In, value, int, RNG seed
      @ Out, None
    """
    self._seed = abs(int(value))
    # According to the numpy docs, best practice is to create a new Generator rather than reseed an
    # existing one.
    bitGenerator = np.random.MT19937()
    # MT19937(seed) doesn't produce the same initial seed and state as the _legacy_seeding method
    # because passing the seed in the MT19937 constructor or the Generator.seed() method creates a
    # SeedSequence object, which effectively hashes the seed value, while _legacy_seeding uses the
    # seed value directly. FIXME: If somebody in the future knows of a way to get the same initial
    # state from the seed without using a private method, please change this! -- j-bryan, 2023-11-09
    bitGenerator._legacy_seeding(self._seed)
    self._engine = np.random.Generator(bitGenerator)

  def getRNGSeed(self):
    """
      Gets the RNG seed
      @ In, None
      @ Out, int, RNG seed value
    """
    return self._seed

  def random(self, size):
    """
      Uses a scaling of Generator.integers because Generator.random() does not return values in the same order
      as CrowRNG.random()
      @ In, size, tuple, size of array to return
      @ Out, vals, np.ndarray, random numbers from RNG engine
    """
    # Generator.random(dtype=np.float32) produces approximately the same outputs as CrowRNG.random(),
    # but the CrowRNG outputs are 64-bit floats calculated as below.
    # NOTE: The 64-bit random number generated here only contains about 32 bits of randomness, since
    # only one state of a 32-bit MT19937 bit generator is used.
    return self._engine.integers(0, 4294967295, endpoint=True, size=size) / 4294967295  # that's 2**32 - 1

  def forwardSeed(self, count):
    """
      Throws away a random number to advance the RNG state
      @ In, count, int, number of random states to progress
      @ Out, None
    """
    self._engine.integers(0, 2 ** 32 - 1, endpoint=True, size=count)

setupCpp()
boxMullerGen = BoxMullerGenerator()
if stochasticEnv == 'numpy':
  npStochEnv = NumpyRNG()
else:
  crowStochEnv = CrowRNG()

def setStochasticEnv(env):
  """
    Sets (or resets) the global stochastic environment

    @ In, env, str, the environment to use; may be 'numpy' or 'crow'
    @ Out, None
  """
  global stochasticEnv, npStochEnv, crowStochEnv, boxMullerGen
  stochasticEnv = env
  if env == 'numpy':
    npStochEnv = NumpyRNG()
  else:
    setupCpp()
    crowStochEnv = CrowRNG()
  boxMullerGen = BoxMullerGenerator()

#
# Utilities
#
#
def randomSeed(value, engine=None):
  """
    Function to get a random seed
    @ In, value, float, the seed
    @ In, engine, instance, optional, random number generator
    @ Out, None
  """
  engine = getEngine(engine)
  engine.seed(value)

def forwardSeed(count, engine=None):
  """
    Function to advance the state of a random number generator engine
    @ In, count, int, number of steps to advance the RNG
    @ In, engine, NumpyRNG or CrowRNG, optional, RNG engine to modify
    @ Out, None
  """
  engine = getEngine(engine)
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
  vals = engine.random(size=(samples, dim))
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
    @ Out, vals, float, random normal number (or np.array with size [n] if n>1, or np.array with size [n,samples] if samples>1)
  """
  engine = getEngine(engine)
  if isinstance(size, int):
    size = (size, )
  vals = boxMullerGen.generate(np.prod(size), engine=engine)
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
  intRange = high - low + 1.0
  rawNum = low + random(engine=engine)*intRange
  rawInt = math.floor(rawNum)
  if rawInt < low or rawInt > high:
    if caller:
      caller.raiseAMessage("Random int out of range")
    rawInt = max(low, min(rawInt, high))
  return rawInt


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
    engine = NumpyRNG()
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

def getEngine(eng=None):
  """
   Choose an engine if it is none and raise error if engine type not recognized
   @ In, engine, instance, optional, random number generator
   @ Out, engine, instance, random number generator
  """
  if eng is None:
    if stochasticEnv == 'numpy':
      eng = npStochEnv
    elif stochasticEnv == 'crow':
      eng = crowStochEnv
  if not isinstance(eng, NumpyRNG) and not isinstance(eng, CrowRNG):
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
