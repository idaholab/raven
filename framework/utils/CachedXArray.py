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
Module to wrap XArray entities with caching utility.
Created on May 22, 2017

@author: talbpaul
"""
from __future__ import division, print_function, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import numpy as np
import xarray as xr

class CachedDataset(object):
  """
    Wrapper class for xarray.Dataset that makes use of entry caching or buffering to mitigate
    the traditional one-at-a-time appending to a dataset.

    Note that there are several options that were considered when defining this class.

    First, we considered extending XR.Dataset by inheriting from it.  This does not work well,
    and is not recommended by the XR documentation at the current time.

    Second, we considered wrapping XR.Dataset, placing an instance of XR.Dataset into a class
    member to access as needed.  Unfortunately, this would drastically reduce the number of
    accessible functions to those we provide our own API to pass through to.

    Finally, we decided on using the "register dataset accessor" method, as this will keep
    intact all the XR.Dataset functionality and still allow us our own custom class.
  """
  # OVERLOADED BASE FUNCTIONS
  def __init__(self,data=None,cacheSize=100,prealloc=False,entries=[]):
    """
      Constructor.  Takes the same arguments as the wrapped class, plus ones specific for caching
      @ In data, xr.Dataset, optional, existing data object to wrap
      @ In, cacheSize, int, optional, length (in objects) of cache
      @ In, prealloc, bool, optional, if True then uses preallocated xarray instead of list of datasets
      @ Out, None
    """
    self.__dataset       = None             # underlying dataset
    self._availableCache = 0                # Next available index of cache for placing buffered data
    self._samples        = 0                # number of recorded samples

    self._cacheSize      = cacheSize        # Acceptable buffer length, has to be set using setCacheSize
    self._prealloc       = prealloc         # switch to flip style FIXME should actually be separate class
    self._entries        = entries          # variables/pointwise metadata variables name list; only needed if preallocating cache

    self._cache          = self.createEmptyCache(cacheSize)  # Buffer of xr.dataSet realizations that can be collapsed onto the main data object

    #depending on "data", construct as we need to
    if data is None:
      pass
    elif isinstance(data,xr.Dataset):
      #assume this is the desired underlying dataset
      self.__dataset = data.copy()
      self._updateDBLen()
    else:
      raise NotImplementedError('Unknown type in CachedXArray construction:',type(data))

  def __str__(self):
    """
      More detailed string representation of object.
      @ In, None
      @ Out, str, str, string representation
    """
    return 'CachedXArray.CachedDataset of '+str(self.__dataset)

  def __len__(self):
    """
      Number of entries in the data.  Looks in "samples" for this.
      @ In, None
      @ Out, len, int, integer length (number of samples stored)
    """
    #try:
    #  merged = len(self.__dataset.sample)
    #except AttributeError:
    #  raise NotImplementedError('"sample" not found when trying to determine length of CachedXArray.CachedDataset!')
    #if self._prealloc:
    #else:
    #  unmerged = sum(1 if x is not None else 0 for x in self._cache)
    unmerged = self._availableCache-1
    return self._samples + unmerged

  # GETTERS AND SETTERS
  def asDataset(self):
    """
      Gives access to the underlying dataset, after flushing
      @ In, None
      @ Out, asDataset, xr.Dataset, dataset
    """
    self.flush()
    return self.__dataset

  def setCacheSize(self,cacheSize):
    """
      Establishes the cache size.  Recommended to only be done once, immediately after initialization.
      @ In cacheSize, int, the length of the buffering array (in number of entries)
      @ Out, None
    """
    # if uninitialized ...
    if self._cache is None:
      self._cache = self.createEmptyCache(cacheSize)
    # else if cache size didn't change, do nothing
    elif cacheSize == self._cacheSize:
      pass
    # else if extending cache size ...
    elif cacheSize > self._cacheSize:
      self.extendCache(cacheSize)
    # else if truncating cache
    else:
      self.shrinkCache(cacheSize)
    # update cache size
    self._cacheSize = cacheSize

  def _updateDBLen(self):
    """
      Updates the useful class variable _samples that stores the number of non-cached entries
      @ In, None
      @ Out, None
    """
    self._samples = len(self.__dataset.sample)

  # UTILITY METHODS
  #@profile
  def append(self,data):
    """
      Caches new data into the buffer.  If this fills the buffer, "flush" the buffer into the data container.
      @ In, data, xarray.Dataset, new realization to add to buffer
                OR dict of rlz values
    """
    # check consistency for future flushing first
    # add data to cache
    #try:
    if self._prealloc:
      for key,val in data.items():
         self._cache[key].loc[{'sample': self._samples + self._availableCache}] = val
    else:
      self._cache[self._availableCache] = data
    #except TypeError as e:
    #  raise TypeError('Tried to "append" in "CachedXArray" but "setCacheSize" not called yet!')

    # increment available cache, and flush if full
    self._availableCache += 1
    if self._availableCache >= self._cacheSize:#len(self._cache):
      self.flush()
  
  #@profile
  def flush(self):
    """
      Flushes the buffered data into the data container.
    """
    if self._prealloc:
      # trim the unused dimensions
      toDel = range(self._samples+self._availableCache,self._samples+self._cacheSize)
      self._cache.drop(toDel,dim='sample')
      # set it to merge
      toMerge = [self._cache]
    else:
      #find the part of the cache that is filled
      firstEmptyIndex = None
      for idx,entry in enumerate(self._cache):
        if entry is None:
          firstEmptyIndex = idx
          break
      if firstEmptyIndex is not None:
        toMerge = self._cache[:firstEmptyIndex]
      else:
        toMerge = self._cache[:]
    if self.__dataset is None:
      self.__dataset = xr.merge(toMerge)
    else:
      self.__dataset = xr.merge([self.__dataset] + toMerge)
    self._updateDBLen()
    self._cache = self.createEmptyCache(self._cacheSize)
    self._availableCache = 0

  def createEmptyCache(self,cacheSize,offset=0):
    """
      Creates an empty cache.
      @ In, cacheSize, int, length of samples
      @ Out, xr.DataSet or list, new cache
    """
    if self._prealloc:
      #sample number
      sampleStart = self._samples
      cacheRange = range(sampleStart+offset,sampleStart+cacheSize+offset)
      empty = dict((v,xr.DataArray(np.zeros(cacheSize),
                                   dims=['sample'],
                                   coords={'sample':cacheRange}))
                                   for v in self._entries)
      return xr.Dataset(data_vars=empty)
    else:
      return [None]*cacheSize

  def extendCache(self,newSize):
    """
      Extend the existing cache to include new space.
      @ In, newSize, int, new size (must be greater than old size)
      @ Out, None
    """
    if self._prealloc:
      old = self._cache.copy()
      new = createEmptyCache(newSize - self._cacheSize, offset = self._cacheSize)
      self._cache = xr.merge([old,new])
    else:
      self._cache = self._cache[:] + [None]*(cacheSize - self._cacheSize)

  def shrinkCache(self,newSize):
    """
      Shrink the existing cache to use less space.  May destroy data.
      @ In, newSize, int, new cache size
      @ Out, None
    """
    # TODO flush before shrinking to preserve data?
    raise NotImplementedError #TODO placeholder, not needed yet
    #self._cache = self._cache[:cacheSize]




class CachedDataArray(CachedDataset):
  def flush(self):
    """
      Flushes the buffered data into the data container.
    """
    #find the part of the cache that is filled
    firstEmptyIndex = None
    for idx,entry in enumerate(self._cache):
      if entry is None:
        firstEmptyIndex = idx
        break
    if firstEmptyIndex is not None:
      toMerge = self._cache[:firstEmptyIndex]
    else:
      toMerge = self._cache[:]
    if self.__dataset is None:
      self.__dataset = xr.concat(toMerge)
    else:
      self.__dataset = xr.concat([self.__dataset] + toMerge)
    self._cache = [None]*self._cacheSize
    self._availableCache = 0


