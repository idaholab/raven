"""
Created on Oct 13, 2015

@author: alfoa
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
#Internal Modules End--------------------------------------------------------------------------------

class ravenArrayIterator(object):
  """
   This class implements a custom RAVEN iterator, that avoids to perform cartesian product for N-Dimensional Grids
  """

  def __init__(self, shape = None, arrayIn = None):
    """
     Init method.
     @ In, shape, tuple or list, (required if 'arrayIn' is not provided), the shape of the N-D array for which an iterator needs to be created
     @ In, arrayIn, ndarray or cachedarray, (required if 'shape' is not provided) the array for which the iterator needs to be created
     @ Out, None
    """
    if shape is None and arrayIn is None        : raise IOError("either shape or arrayIn need to be passed in")
    if shape is not None and arrayIn is not None: raise IOError("both shape or arrayIn are passed in")
    self.shape     = shape if shape is not None else arrayIn.shape
    self.ndim      = len(self.shape)
    self.maxCnt    = np.prod(self.shape)
    self.cnt       = 0
    self.finished  = False
    self.iterator  = None

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
        for i in xrange(len(self.iterator)-1, -1, -1):

          if self.iterator[i] + 1 >= self.shape[i]:
            self.iterator[i] = 0
            continue
          else:
            self.iterator[i]+=1
            break
    else:
      self.iterator.iternext()
      self.finished = self.iterator.finished
      if not self.finished: self.multiIndex = self.iterator.multi_index

    return self.finished

  def reset(self):
    """
     This method resets the iterator to its initial status
     @ In, None
     @ Out, None
    """
    self.cnt, self.finished = 0, False
    if type(self.iterator).__name__ == 'list': self.iterator = [0]*self.ndim
    else:
      self.iterator.reset()
      self.multiIndex, self.finished = self.iterator.multi_index, self.iterator.finished

  def __iter__(self):
    """
     Methot that returns the iterator object and is implicitly called at the start of loops.
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
    if self.finished: raise StopIteration
    if type(self.iterator).__name__ == 'list': return self.iterator
    else                                     : return self.iterator.multi_index




