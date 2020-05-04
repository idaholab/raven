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
Created on Oct 13, 2015

@author: alfoa
"""
from __future__ import division, print_function, unicode_literals, absolute_import
#External Modules------------------------------------------------------------------------------------
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
#Internal Modules End--------------------------------------------------------------------------------

class ravenArrayIterator(object):
  """
   This class implements a custom RAVEN iterator, that avoids creating the cartesian product for N-Dimensional Grids
  """
  def __init__(self, shape = None, arrayIn = None):
    """
     Init method.
     @ In, shape, tuple or list, (required if 'arrayIn' is not provided), the shape of the N-D array for which an iterator needs to be created
     @ In, arrayIn, ndarray or cachedarray, (required if 'shape' is not provided) the array for which the iterator needs to be created
     @ Out, None
    """
    if shape is None and arrayIn is None:
      raise IOError("either shape or arrayIn need to be passed in")
    if shape is not None and arrayIn is not None:
      raise IOError("both shape and arrayIn are passed in")
    self.shape     = shape if shape is not None else arrayIn.shape # array shape. tuple(stepsInDim1, stepsInDim2,....stepsInDimN)
    self.ndim      = len(self.shape)                               # number of dimension
    self.maxCnt    = np.prod(self.shape)                           # maximum number of combinations
    self.cnt       = 0                                             # counter used for the iterations
    self.finished  = False                                         # has the iterator hitted the end?
    self.iterator  = None                                          # this is the internal iterator object.
                                                                   # if the variable 'shape' is passed in, the iterator is going to be created internally (no cartesian product needed)
                                                                   # if the variable 'arrayIn' is passed in, the iterator is going to be associated to the numpy.nditer (it requires the
                                                                   #                                                 cartesian product, since the arrayIn needs to represent a full grid)
    if arrayIn is not None:
      self.iterator   = np.nditer(arrayIn,flags=['multi_index'])
      self.multiIndex = self.iterator.multi_index
    else:
      self.iterator   = [0]*self.ndim
      self.multiIndex = self.iterator

  def iternext(self):
    """
      This method checks whether iterations are left, and perform a
      single internal iteration without returning the result.
      @ In, None
      @ Out, self.finished, bool, return if the iteration finished
    """
    self.cnt += 1
    #if self.cnt != 1:
    if type(self.iterator).__name__ == 'list':
      if self.cnt >= self.maxCnt:
        self.finished = True
      else:
        for i in range(len(self.iterator)-1, -1, -1):
          if self.iterator[i] + 1 >= self.shape[i]:
            self.iterator[i] = 0
            continue
          else:
            self.iterator[i]+=1
            break
    else:
      self.iterator.iternext()
      self.finished = self.iterator.finished
      if not self.finished:
        self.multiIndex = self.iterator.multi_index
    return self.finished

  def reset(self):
    """
      This method resets the iterator to its initial status
      @ In, None
      @ Out, None
    """
    self.cnt, self.finished = 0, False
    if type(self.iterator).__name__ == 'list':
      self.iterator = [0]*self.ndim
    else:
      self.iterator.reset()
      self.multiIndex, self.finished = self.iterator.multi_index, self.iterator.finished

  def __iter__(self):
    """
      Method that returns the iterator object and is implicitly called at the start of loops.
      @ In, None
      @ Out, self, iterator object
    """
    return self

  def next(self):
    """
      Method to get the next pointer (value), if finished, raise a StopIteration exception
      @ In, None
      @ Out, iterator, tuple, the n-d iterator
    """
    return self.__next__()

  def __next__(self):
    """
      See next(self)
      @ In, None
      @ Out, iterator, tuple, the n-d iterator
    """
    self.iternext()
    if self.finished:
      raise StopIteration
    if type(self.iterator).__name__ == 'list':
      return self.iterator
    else:
      return self.iterator.multi_index
