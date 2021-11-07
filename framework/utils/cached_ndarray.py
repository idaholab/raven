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
Created on Feb 4, 2015

@author: alfoa
"""
#----- python 2 - 3 compatibility
from __future__ import division, print_function, absolute_import
#----- end python 2 - 3 compatibility
#External Modules------------------------------------------------------------------------------------
import sys
import threading
from numpy import ndarray
import numpy as np
lock = threading.Lock()
#External Modules End--------------------------------------------------------------------------------

class c1darray(object):
  """
    This class performs the caching of the numpy ndarray class
  """
  def __init__(self, shape = (100,), values = None, dtype=float, buff=None, offset=0, strides=None, order=None):
    """
      Constructor
      @ In, shape, tuple, optional, array shape
      @ In, values, numpy.ndarray, optional, array through which this c1darray needs to be initialized
      @ In, dtype, np.type, optional, the data type of this array
      @ In, buff, int, optional, buffer size
      @ In, offset, int, optional, array offeset
      @ In, strides, object, optional, strides (see numpy)
      @ In, order, string, optional, array ordering (fortran, c, etc) (see numpy)
      @ Out, None
    """
    if values is not None:
      if shape != (100,) and values.shape != shape:
        raise IOError("different shape")
      if type(values).__name__ != 'ndarray':
        raise IOError("Only ndarray is accepted as type.Got "+type(values).__name__)
      self.values = values
      self.size = values.size
    else:
      self.values = ndarray(shape, dtype, buff, offset, strides, order)
      self.size = 0
    try:
      self.capacity = self.values.shape[0]
    except IndexError:
      self.capacity = []
    self.ndim = self.values.ndim

  def __iter__(self):
    """
      Overload of iterator
      @ In, None
      @ Out, __iter__, iterator, iterator
    """
    return self.values[:self.size].__iter__()

  def __getitem__(self, val):
    """
      Get item method. slicing available:
      example 1. c1darrayInstance[5], is going to return the 6th element in  the array
      example 2. c1darrayInstance[1:3], is going to return an array composed by the 2nd,3rd,4th elements
      @ In, val, slice object, the slicing object (e.g. 1, :, :2, 1:3, etc.)
      @ Out, __getitem__, array slicing, the element or the slicing
    """
    return self.values[:self.size].__getitem__(val)

  def __len__(self):
    """
      Return size
      @ In, None
      @ Out, self.size, integer, size
    """
    return self.size

  def append(self, x):
    """
      Append method. call format c1darrayInstance.append(value)
      @ In, x, element or array, the value or array to append
      @ Out, None (appending in place)
    """

    #lock.acquire()
    try:
      if type(x).__name__ not in ['ndarray', 'c1darray']:
        if self.size  == self.capacity:
          self.capacity *= 4
          newdata = np.zeros((self.capacity,),dtype=self.values.dtype)
          newdata[:self.size] = self.values[:]
          self.values = newdata
        self.values[self.size] = x
        self.size  += 1
      else:
        if (self.capacity - self.size) < x.size:
          # to be safer
          self.capacity += max(self.capacity*4, x.size) #self.capacity + x.size*4
          newdata = np.zeros((self.capacity,),dtype=self.values.dtype)
          newdata[:self.size] = self.values[:self.size]
          self.values = newdata
        #for index in range(x.size):
        self.values[self.size:self.size+x.size] = x[:]
        self.size  += x.size
    finally:
      #lock.release()
      pass

  def returnIndexClosest(self,value):
    """
      Function that return the index of the element in the array closest to value
      @ In, value, double, query value
      @ Out, index, int, index of the element in the array closest to value
    """
    index=-1
    dist = sys.float_info.max
    for i in range(self.size):
      if abs(self.values[i]-value)<dist:
        dist = abs(self.values[i]-value)
        index = i
    return index

  def returnIndexFirstPassage(self,value):
    """
      Function that return the index of the element that firstly crosses value
      @ In, value, double, query value
      @ Out, index, int, index of the element in the array closest to value
    """
    index=-1
    dist = sys.float_info.max
    for i in range(1,self.size):
      if (self.values[i]>=value and self.values[i-1]<=value) or (self.values[i]<=value and self.values[i-1]>=value):
        index = i
        break
    return index

  def returnIndexMax(self):
    """
      Function that returns the index (i.e. the location) of the maximum value of the array
      @ In, None
      @ Out, index, int, index of the maximum value of the array
    """
    index=-1
    maxValue = -sys.float_info.max
    for i in range(self.size):
      if self.values[i]>=maxValue:
        maxValue = self.values[i]
        index = i
        #break Breaking here guarantees you only ever get the first index (unless you have -sys.float_info_max in first entry)
    return index

  def returnIndexMin(self):
    """
      Function that returns the index (i.e. the location) of the minimum value of the array
      @ In, None ,
      @ Out, index, int, index of the minimum value of the array
    """
    index=-1
    minValue = sys.float_info.max
    for i in range(self.size):
      if self.values[i]<=minValue:
        minValue = self.values[i]
        index = i
        #break Breaking here guarantees you only ever get the first index (unless you have sys.float_info_max in first entry)
    return index

  def __add__(self, x):
    """
      Method to mimic the addition of two arrays
      @ In, x, c1darray, the addendum
      @ Out, newArray, c1drray, sum of the two arrays
    """
    newArray = c1darray(shape = self.size+np.array(x).shape[0], values=self.values[:self.size]+np.array(x))
    return newArray

  def __radd__(self, x):
    """
      reversed-order (LHS <-> RHS) addition
      Method to mimic the addition of two arrays
      @ In, x, c1darray, the addendum
      @ Out, newArray, c1drray, sum of the two arrays
    """
    newArray = c1darray(shape = np.array(x).shape[0]+self.size, values=np.array(x)+self.values[:self.size])
    return newArray

  def __array__(self, dtype = None):
    """
      so that numpy's array() returns values
      @ In, dtype, np.type, the requested type of the array
      @ Out, __array__, numpy.ndarray, the requested array
    """
    if dtype != None:
      return ndarray((self.size,), dtype, buffer=None, offset=0, strides=None, order=None)
    else            :
      return self.values[:self.size]

  def __repr__(self):
    """
      overload of __repr__ function
      @ In, None
      @ Out, __repr__, string, the representation string
    """
    return repr(self.values[:self.size])

#
#
#
#
class cNDarray(object):
  """
    Higher-dimension caching of numpy arrays.  Might include c1darray as a subset if designed right.

    DEV NOTE:
    When redesigning the DataObjects in RAVEN in 2017, we tried a wide variety of libraries, strategies,
    and data structures.  For appending one realization (with N entities) at a time, the np.ndarray proved
    most efficient for dropping in values, particularly when cached as per this class.  Restructuring the data
    into a more useful form (e.g. xarray.Dataset) should be accomplished in the DataObject; this is just a collecting
    structure. - talbpw, 2017-10-20
  """
  ### CONSTRUCTOR ###
  def __init__(self,values=None,width=None,length=None,dtype=float,buff=None,offset=0,strides=None,order=None):
    """
      Constructor.
      @ In, values, np.ndarray, optional, matrix of initial values with shape (# samples, # entities)
      @ In, width, int, optional, if not using "values" then this is the number of entities to allocate
      @ In, length, int, optional, if not using "values" then this is the initial capacity (number of samples) to allocate
      @ In, dtype, type, optional, sets the type of the content of the array
      @ In, buff, int, optional, buffer size
      @ In, offset, int, optional, array offeset
      @ In, strides, object, optional, strides (see docs for np.ndarray)
      @ In, order, string, optional, array ordering (fortran, c, etc) (see docs for np.ndarray)
      @ Out, None
    """
    # members of this class
    self.values   = None   # underlying data for this structure, np.ndarray with optional dtype (default float)
    self.size     = None   # number of rows (samples) with actual data (not including empty cached)
    self.width    = None   # number of entities aka columns
    self.capacity = None   # cached np.ndarray size
    # priorities: initialize with values; if not, use width and length
    if values is not None:
      if type(values) != np.ndarray:
        raise IOError('Only np.ndarray can be used to set "values" in "cNDarray".  Got '+type(values).__name__)
      self.values = values         # underlying data structure
      self.size = values.shape[0]
      try:
        self.width = values.shape[1]
      except IndexError:
        ## TODO NEEDS TO BE DEPRECATED should always have a width, in real usage
        self.width = 0
      # if setting by value, initialize capacity to existing data length
      self.capacity = self.size
    else:
      if width is None:
        raise IOError('Creating cNDarray: neither "values" nor "width" was specified!')
      self.capacity = length if length is not None else 100
      self.width = width
      self.size = 0
      self.values = ndarray((self.capacity,self.width),dtype,buff,offset,strides,order)

  ### PROPERTIES ###
  @property
  def shape(self):
    """
      Shape property, as used in np.ndarray structures.
      @ In, None
      @ Out, (int,int), the (#rows, #columns) of useful data in this cached array
    """
    return (self.size,self.width)

  ### BUILTINS ###
  def __array__(self, dtype = None):
    """
      so that numpy's array() returns values
      @ In, dtype, np.type, the requested type of the array
      @ Out, __array__, numpy.ndarray, the requested array
    """
    if dtype != None:
      return ndarray((self.size,self.width), dtype, buffer=None, offset=0, strides=None, order=None)
    else:
      return self.getData()

  def __getitem__(self,val):
    """
      Get item method.  Slicing should work as expected.
      @ In, val, slice object, the slicing object (e.g. 1, :, :2, 1:3, etc.)
      @ Out, __getitem__, np.ndarray, the element(s)
    """
    return self.values[:self.size].__getitem__(val)

  def __iter__(self):
    """
      Overload of iterator
      @ In, None
      @ Out, __iter__, iterator, iterator
    """
    return self.values[:self.size].__iter__()

  def __len__(self):
    """
      Return size, which is the number of samples, independent of entities, containing useful data.
      Does not include cached entries that have not yet been filled.
      @ In, None
      @ Out, __len__, integer, size
    """
    return self.size

  def __repr__(self):
    """
      overload of __repr__ function
      @ In, None
      @ Out, __repr__, string, the representation string
    """
    return repr(self.values[:self.size])

  ### UTILITY FUNCTIONS ###
  def append(self,entry):
    """
      Append method. call format c1darrayInstance.append(value)
      @ In, entry, np.ndarray, the entries to append as [entry, entry, entry].  Must have shape (x, # entities), where x can be any nonzero number of samples.
      @ Out, None
    """
    # TODO extend to include sending in a (width,) shape np.ndarray to append a single sample, rather than have it forced to be a 1-entry array.
    # entry.shape[0] is the number of new entries, entry.shape[1] is the number of variables being entered
    # entry must match width and be at least 1 entry long
    if type(entry) not in [np.ndarray]:
      raise IOError('Tried to add new data to cNDarray.  Can only accept np.ndarray, but got '+type(entry).__name__)
    # for now require full correct shape, later handle the single entry case
    if len(entry.shape)!=1:
      # TODO single entry case
      raise IOError('Tried to add new data to cNDarray.  Need shape ({},) but got "{}"!'.format(self.width,entry.shape))
    # must have matching width (fix for single entry case)
    if entry.shape[0] != self.width:
      raise IOError('Tried to add new data to cNDarray.  Need {} entries in array, but got '.format(self.width)+str(entry.shape[0]))
    # check if there's enough space in cache to append the new entries
    if self.size + 1 > self.capacity:
      # since there's not enough space, quadruple available space # TODO change growth parameter to be variable?
      self.capacity += self.capacity*3
      newdata = np.zeros((self.capacity,self.width),dtype=self.values.dtype)
      newdata[:self.size] = self.values[:self.size]
      self.values = newdata
    self.values[self.size] = entry[:]
    self.size += 1

  def addEntity(self,vals,firstEver=False):
    """
      Adds a column to the dataset.
      @ In, vals, list, as list(#,#,#) where # is either single-valued or numpy array
      @ Out, None
    """
    # create a new column with up to the cached capacity
    new = np.ndarray(self.capacity,dtype=object)
    # fill up to current filled size with the values
    new[:self.size] = vals
    # reshape so it can be stacked onto the existing data
    new = new.reshape(self.capacity,1)
    # "hstack" stacks along the second dimension, or columns for us
    self.values = np.hstack((self.values,new))
    self.width += 1

  def getData(self):
    """
      Returns the underlying data structure.
      @ In, None
      @ Out, getData, np.ndarray, underlying data up to the used size
    """
    return self.values[:self.size]

  def removeEntity(self,index):
    """
      Removes a column from this dataset
      @ In, index, int, index of entry to remove
      @ Out, None
    """
    assert(abs(index) < self.width)
    self.values = np.delete(self.values,index,axis=1)
    self.width -= 1
