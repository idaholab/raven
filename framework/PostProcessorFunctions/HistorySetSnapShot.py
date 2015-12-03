"""
Created on October 28, 2015

"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import copy
import itertools
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from cached_ndarray import c1darray
from .Data import Data, NotConsistentData, ConstructError
import utils
import DataObjects
#Internal Modules End--------------------------------------------------------------------------------

def HistorySetSampling(inputDict,samplingType):
  """
   This function does things.
   @ In : inputDict
   @ Out: outputDict
  """

  outputDict = copy.deepcopy(inputDict)

  return outputDict

