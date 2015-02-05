"""
Created on Feb 4, 2015

@author: alfoa
"""
from numpy import ndarray
import numpy as np

class c1darray(object):
  """
  This class performs the caching of the numpy ndarray class
  """
  def __init__(self, shape = (100,), values = None, dtype=float, buff=None, offset=0, strides=None, order=None):
    """
    Constructor
    """
    if values != None:
      if shape != (100,) and values.shape != shape: raise IOError("different shape")
      if type(values).__name__ != 'ndarray': raise IOError("Only ndarray is accepted as type.Got "+type(values).__name__)
      self.values = values
      self.size = values.size
    else: 
      self.values = ndarray(shape, dtype, buff, offset, strides, order)
      self.size = 0
    self.capacity = self.values.shape[0]

  def __getitem__(self, val): 
    """
      Get item method. slicing available:
      example 1. c1darrayInstance[5], is going to return the 6th element in  the array
      example 2. c1darrayInstance[1:3], is going to return an array composed by the 2nd,3rd,4th elements
      @ In, slice object, the slicing object (e.g. 1, :, :2, 1:3, etc.)
      @ Out, the element or the slicing
    """
    return self.values[:self.size].__getitem__(val)
   
  def __len__(self): 
    """
     Return size
     @ In, None
     @ Out, integer, size
    """
    return self.size
  
  def append(self,x):
    """
    Append method. call format c1darrayInstance.append(value)
    @ In, element or array, the value or array to append
    @ Out, None, appending in place
    """
    if type(x).__name__ != 'ndarray':
      if self.size  == self.capacity:
        self.capacity *= 4
        newdata = np.zeros((self.capacity,))
        newdata[:self.size] = self.values
        self.values = newdata
      self.values[self.size] = x
      self.size  += 1
    else:
      if (self.capacity - self.size) < x.size:
        # to be safer
        self.capacity += self.capacity + x.size
        newdata = np.zeros((self.capacity,))
        newdata[:self.size] = self.values
        self.values = newdata
      for index in range(x.size):
        self.values[self.size] = x[index]
        self.size  += 1
  
  def __add__(self, x):
    """ 
    addition
    """
    x = np.array(x) # make sure input is numpy compatible
    return c1darray(shape = self.values[:self.size].shape[0]+x.shape[0], values=self.values[:self.size]+x)

  def __radd__(self, x):
    """ reversed-order (LHS <-> RHS) addition
    """
    x = np.array(x) # make sure input is numpy compatible
    return c1darray(shape = x.shape[0]+self.values[:self.size].shape[0], values=x+self.values[:self.size])

  def __array__(self):
    """ so that numpy's array() returns values
    """
    return self.values[:self.size]

  def __repr__(self):
    return "< c1darray. Masked size: " +str(self.size)+". Real size: "+str(self.capacity)+". Object: "+repr(self.values[:self.size])+" >"
