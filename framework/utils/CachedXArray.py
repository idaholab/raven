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
  def __init__(self,data=None,cacheSize=100):
    """
      Constructor.  Takes the same arguments as the wrapped class, plus ones specific for caching
      @ In data, xr.Dataset, existing data object to wrap
      @ In, cacheSize, int, length (in objects) of cache
      @ Out, None
    """
    self.__dataset       = None             # underlying dataset
    self._cacheSize      = cacheSize        # Acceptable buffer length, has to be set using setCacheSize
    self._cache          = [None]*cacheSize # Buffer of xr.dataSet realizations that can be collapsed onto the main data object
    self._availableCache = 0                # Next available index of cache for placing buffered data

    #depending on "data", construct as we need to
    if data is None:
      pass
    elif isinstance(data,xr.Dataset):
      #assume this is the desired underlying dataset
      self.__dataset = data
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
    try:
      merged = len(self.__dataset.sample)
    except AttributeError:
      raise NotImplementedError('"sample" not found when trying to determine length of CachedXArray.CachedDataset!')
    unmerged = sum(1 if x is not None else 0 for x in self._cache)
    return merged + unmerged

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
      self._cache = [None]*cacheSize
    # else if cache size didn't change, do nothing
    elif cacheSize == self._cacheSize:
      pass
    # else if extending cache size ...
    elif cacheSize > self._cacheSize:
      self._cache = self._cache[:] + [None]*(cacheSize - self._cacheSize)
    # else if truncating cache
    else:
      self._cache = self._cache[:cacheSize]
    # update cache size
    self._cacheSize = cacheSize

  # UTILITY METHODS
  def append(self,data):
    """
      Caches new data into the buffer.  If this fills the buffer, "flush" the buffer into the data container.
      @ In, data, xarray.Dataset, new realization to add to buffer
    """
    # check consistency for future flushing first
    # add data to cache
    try:
      self._cache[self._availableCache] = data
    except TypeError as e:
      raise TypeError('Tried to "append" in "CachedXArray" but "setCacheSize" not called yet!')

    # increment available cache, and flush if full
    self._availableCache += 1
    if self._availableCache >= len(self._cache):
      self.flush()

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
      self.__dataset = xr.merge(toMerge)
    else:
      self.__dataset = xr.merge([self.__dataset] + toMerge)
    self._cache = [None]*self._cacheSize
    self._availableCache = 0

