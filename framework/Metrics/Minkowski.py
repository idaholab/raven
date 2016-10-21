"""
Created on Jul 18 2016

@author: mandd
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import os
import shutil
import math
import numpy as np
import abc
import importlib
import inspect
import atexit
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from Assembler import Assembler
import CustomCommandExecuter
import utils
import mathUtils
import TreeStructure
import Files
from .Metric import Metric

#Internal Modules End--------------------------------------------------------------------------------

class Minkowski(Metric):

  def initialize(self,inputDict):
    self.p = None
    self.timeID = None

  def _readMoreXML(self,xmlNode):
    for child in xmlNode:
      if child.tag == 'p':
        self.p = float(child.text)
      if child.tag == 'timeID':
        self.timeID = child.text

  def distance(self,x,y):
    if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
      value = 0
      for i in range(x.size):
        value += (math.abs(x[i]-y[i]))**self.p
      return math.pow(value,1/p)
    elif isinstance(x,dict) and isinstance(y,dict):
      if self.timeID == None:
        self.raiseAnError(IOError,'The Minkowski metrics is being used on a historySet without the parameter timeID being specified')
      if x.keys() == y.keys():
        value = 0
        for key in x.keys():
          if x[key].size == y[key].size:
            if key == self.timeID:
              for i in range(x[key].size):
                value += (math.abs(x[i]-y[i]))**self.p
            return math.pow(value,1/p)
          else:
            print('Metric Minkowski error: the length of the variable array ' + str(key) +' is not consistent among the two data sets')
      else:
        print('Metric Minkowski error: the two data sets do not contain the same variables')
    else:
      print('Metric Minkowski error: the structures of the two data sets are different')
