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

import copy
import numpy as np
from collections import deque

from utils.utils import find_distribution1D

# in general, we will use Crow for now, but let's make it easy to switch just in case it is helpfull eventually.
stochasticEnv = 'crow'
#stochasticEnv = 'numpy'

# internal utilities
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


# [0,1] to Normal converter
class BoxMullerGenerator:
  """
    Iterator class for the Box-Muller transform
  """
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.queue = deque()

  def generate(self):
    """
      Yields a normally-distributed pseudorandom value
      @ In, None
      @ Out, generate, float, random value
    """
    if len(self.queue) == 0:
      #calculate new values
      self.queue.extend(self.createSamples())
    return self.queue.pop() #no need to pop left, as they're independent and all get used

  def createSamples(self):
    """
      Sample calculator.  Because Box Muller does batches of 2, add them to a queue.
      @ In, None
      @ Out, (z1,z2), tuple, two independent random values
    """
    u1,u2 = random(2)
    z1 = np.sqrt(-2.*np.log(u1))*np.cos(2.*np.pi*u2)
    z2 = np.sqrt(-2.*np.log(u1))*np.sin(2.*np.pi*u2)
    return z1,z2

  def testSampling(self, n=1e5):
    """
      Tests distribution of samples over a large number.
      @ In, n, int, optional, number of samples to test with
      @ Out, mean, float, mean of sample set
      @ Out, stdev, float, standard deviation of sample set
    """
    n = int(n)
    samples = np.array([self.generate() for _ in range(n)])
    mean = np.average(samples)
    stdev = np.std(samples)
    return mean,stdev

distribution1D = find_distribution1D()
crowStochEnv = distribution1D.DistributionContainer.instance()
boxMullerGen = BoxMullerGenerator()

#### RNG CLASS ####
# TODO make the global RNG a module-level object to reduce duplication
class CrowRNG: # TODO message user?
  """
    For entities that need to rely on a self-controlled random number generator instead
    of using the global RNG, this tool is offered.
  """
  def __init__(self,seed=None,messageHandler=None,freshEngine=False):
    """
      Constructor.
      @ In, seed, int, optional, if provided then RNG will use the specified seed.  If not specified, the next global random int will be used.
      @ In, messageHandler, MessageHandler, optional, should be provided for message tracking if at all possible
      @ In, freshEngine, bool, optional, if True then get a clean instance of the RNG engine for this entity
      @ Out, None
    """
    # class members
    self.messageHandler      = None # message handling TODO class needs to inherit from message user!
    self.engine              = None # stochastic engine, from crow
    self.lastSeed            = None # tracks the last seed used to initialize RNG
    self.count               = 0    # tracks number of uses since seed last changed
    # algorithmic initialization
    ## hold the message handler at class level
    self.messageHandler = messageHandler
    ## set the stochastic generator -> NOTE by default we get a pointer, not an independent copy!
    self.__setEngine(freshEngine)
    ## get a seed if one wasn't provided # TODO someday do this for more control; for now don't b/c it would fail many regression tests -> find seed?
    #if seed is None:
    #  seed = 1256955321 # from crow.include.distributions.distribution.h, _defaultSeed
    #  #seed = self.randomIntegers(0,2**31)
    ## track the last seed used
    self.lastSeed = seed
    ## set the seed
    if seed is not None:
      self.setSeed(seed)

  def __getstate__(self):
    """
      Obtains state of object for pickling.
      @ In, None
      @ Out, d, dict, stateful dictionary
    """
    d = dict((key,copy.deepcopy(val)) for key,val in self.__dict__.items() if key not in ['engine'])
    # "engine" is unpicklable, so save state instead with self.lastSeed and self.count
    #del d['engine']
    return d

  def __setstate__(self,d):
    """
      Sets the state of the object for unpickling
      @ In, d, dict, stateful dictionary
      @ Out, None
    """
    # default setstate implementation
    self.__dict__ = d
    # set up unpicklable stochastic engine
    self.__setEngine()
    # set seed and move forward to the right count
    self.setSeed(self.lastSeed,self.count)

  def random(self,dim=1,samples=1,keepMatrix=False):
    """
      Function to get a single random value, an array of random values, or a matrix of random values, on [0,1]
      @ In, dim, int, optional, dimensionality of samples
      @ In, samples, int, optional, number of arrays to deliver
      @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
      @ Out, vals, float, random normal number (or np.array with size [n] if n>1, or np.array with size [n,samples] if sampels>1)
    """
    dim = int(dim)
    samples = int(samples)
    vals = np.zeros([samples,dim])
    for i in range(len(vals)):
      for j in range(len(vals[0])):
        vals[i][j] = self.engine.random()
        self.count += 1
    if keepMatrix:
      return vals
    else:
      return _reduceRedundantListing(vals,dim,samples)

  def randomIntegers(self,low,high):
    """
      Function to get a random integer
      @ In, low, int, low boundary
      @ In, high, int, upper boundary
      @ Out, rawInt, int, random int
    """
    intRange = high-low
    rawNum = low + self.random()*intRange
    rawInt = int(round(rawNum))
    if rawInt < low or rawInt > high:
      self.raiseAMessage("Random int out of range")
      rawInt = max(low,min(rawInt,high))
    return rawInt

  def randomNormal(self,dim=1,samples=1,keepMatrix=False):
    """
      Function to get a single random value, an array of random values, or a matrix of random values, normally distributed
      @ In, dim, int, optional, dimensionality of samples
      @ In, samples, int, optional, number of arrays to deliver
      @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
      @ Out, vals, float, random normal number (or np.array with size [n] if n>1, or np.array with size [n,samples] if sampels>1)
    """
    dim = int(dim)
    samples = int(samples)
    vals = np.zeros([samples,dim])
    for i in range(len(vals)):
      for j in range(len(vals[0])):
        vals[i,j] = boxMullerGen.generate()
    if keepMatrix:
      return vals
    else:
      return _reduceRedundantListing(vals,dim,samples)

  def randomPermutation(self,l):
    """
      Function to get a random permutation
      @ In, l, list, list to be permuted
      @ Out, newList, list, randomly permuted list
    """
    newList = []
    oldList = l[:]
    while len(oldList) > 0:
      newList.append(oldList.pop(self.randomIntegers(0,len(oldList)-1)))
    return newList

  def randPointsInHypersphere(self,dim,samples=1,r=1,keepMatrix=False):
    """
      obtains a random point internal to a hypersphere of dimension "n" with radius "r"
      see http://www.sciencedirect.com/science/article/pii/S0047259X10001211
      "On decompositional algorithms for uniform sampling from n-spheres and n-balls", Harman and Lacko, 2010, J. Multivariate Analysis
      @ In, dim, int, the dimensionality of the hypersphere
      @ In, r, float, the radius of the hypersphere
      @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
      @ Out, pt, np.array(float), a random point on the surface of the hypersphere
    """
    #sample on surface of n+2-sphere and discard the last two dimensions
    pts = self.randPointsOnHypersphere(dim+2,samples=samples,r=r,keepMatrix=True)[:,:-2]
    if keepMatrix:
      return pts
    else:
      return _reduceRedundantListing(pts,dim,samples)
    return pts

  def randPointsOnHypersphere(self,dim,samples=1,r=1,keepMatrix=False):
    """
      obtains random points on the surface of a hypersphere of dimension "n" with radius "r".
      see http://www.sciencedirect.com/science/article/pii/S0047259X10001211
      "On decompositional algorithms for uniform sampling from n-spheres and n-balls", Harman and Lacko, 2010, J. Multivariate Analysis
      @ In, dim, int, the dimensionality of the hypersphere
      @ In, samples, int, optional, the number of samples desired
      @ In, r, float, optional, the radius of the hypersphere
      @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
      @ Out, pts, np.array(np.array(float)), random points on the surface of the hypersphere [sample#][pt]
    """
    ## first fill random samples
    pts = self.randomNormal(dim,samples=samples,keepMatrix=True)
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

  def __setEngine(self,new=True):
    """
      Establishes an engine for RNG operations
      SHOULD NOT BE CALLED outside of __init__ or __setstate__; otherwise, seed and place need to be tracked carefully
      @ In, new, bool, optional, if True then get a fresh instance of the DistributionContainer (otherwise get a pointer to the shared copy)
      @ Out, None
    """
    # NOTE passing "True" means each time __setEngine is called, an independent DistributionContainer will be created.
    self.engine = distribution1D.DistributionContainer.instance(True)

  def setSeed(self,value,n=0):
    """
      Function to set a random seed
      @ In, value, int, the seed
      @ Out, None
    """
    print('RNG: setting seed to "{}" with shift "{}"'.format(value,n))
    crowStochEnv.seedRandom(value,n)
    self.count = n
    self.lastSeed = value

#### GLOBAL METHODS ####
# For entities using the global seed, the following methods are available
## These are all pass-throughs to a global instance of the RNG creator for now.
print('randomUtils: Instantiating global RNG ...')
globalRNG = CrowRNG()

def randomSeed(value):
  """
    Function to get a random seed
    @ In, value, float, the seed
    @ Out, None
  """
  print('randomUtils: Global random number seed has been changed to',value)
  return globalRNG.setSeed(value)

def random(dim=1,samples=1,keepMatrix=False):
  """
    Function to get a single random value, an array of random values, or a matrix of random values, on [0,1]
    @ In, dim, int, optional, dimensionality of samples
    @ In, samples, int, optional, number of arrays to deliver
    @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
    @ Out, vals, float, random normal number (or np.array with size [n] if n>1, or np.array with size [n,samples] if sampels>1)
  """
  return globalRNG.random(dim=dim,samples=samples,keepMatrix=keepMatrix)

def randomNormal(dim=1,samples=1,keepMatrix=False):
  """
    Function to get a single random value, an array of random values, or a matrix of random values, normally distributed
    @ In, dim, int, optional, dimensionality of samples
    @ In, samples, int, optional, number of arrays to deliver
    @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
    @ Out, vals, float, random normal number (or np.array with size [n] if n>1, or np.array with size [n,samples] if sampels>1)
  """
  return globalRNG.randomNormal(dim=dim,samples=samples,keepMatrix=keepMatrix)

def randomIntegers(low,high,caller):
  """
    Function to get a random integer
    @ In, low, int, low boundary
    @ In, high, int, upper boundary
    @ In, caller, instance, object requesting the random integers
    @ Out, rawInt, int, random int
  """
  return globalRNG.randomIntegers(low,high)

def randomPermutation(l,caller):
  """
    Function to get a random permutation
    @ In, l, list, list to be permuted
    @ In, caller, instance, the caller
    @ Out, newList, list, randomly permuted list
  """
  return globalRNG.randomPermutation(l)

def randPointsOnHypersphere(dim,samples=1,r=1,keepMatrix=False):
  """
    obtains random points on the surface of a hypersphere of dimension "n" with radius "r".
    see http://www.sciencedirect.com/science/article/pii/S0047259X10001211
    "On decompositional algorithms for uniform sampling from n-spheres and n-balls", Harman and Lacko, 2010, J. Multivariate Analysis
    @ In, dim, int, the dimensionality of the hypersphere
    @ In, samples, int, optional, the number of samples desired
    @ In, r, float, optional, the radius of the hypersphere
    @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
    @ Out, pts, np.array(np.array(float)), random points on the surface of the hypersphere [sample#][pt]
  """
  return globalRNG.randPointsOnHypersphere(dim,samples=samples,r=r,keepMatrix=keepMatrix)

def randPointsInHypersphere(dim,samples=1,r=1,keepMatrix=False):
  """
    obtains a random point internal to a hypersphere of dimension "n" with radius "r"
    see http://www.sciencedirect.com/science/article/pii/S0047259X10001211
    "On decompositional algorithms for uniform sampling from n-spheres and n-balls", Harman and Lacko, 2010, J. Multivariate Analysis
    @ In, dim, int, the dimensionality of the hypersphere
    @ In, r, float, the radius of the hypersphere
    @ In, keepMatrix, bool, optional, if True then will always return np.array(np.array(float))
    @ Out, pt, np.array(float), a random point on the surface of the hypersphere
  """
  return globalRNG.randPointsInHypersphere(dim=dim,samples=samples,r=r,keepMatrix=keepMatrix)

