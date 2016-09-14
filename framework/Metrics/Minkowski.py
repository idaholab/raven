#External Modules------------------------------------------------------------------------------------
import os
import shutil
import math
import numpy as np
import abc
import importlib
import inspect
import atexit
import scipy.spatial.distance as spDist
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
      value = spDist.minkowski(x, y, self.p)     
      return value
    elif isinstance(x,dict) and isinstance(y,dict):
      if self.timeID == None:
        self.raiseAnError(IOError,'The Minkowski metrics is being used on a historySet without the parameter timeID being specified')
      if x.keys() == y.keys():
        value = 0
        for key in x.keys():
          if x[key].size == y[key].size:
            if key != self.timeID:
              value += spDist.minkowski(x[key], y[key], self.p)
              '''for i in range(x[key].size):
                value += (abs(x[key][i]-y[key][i]))**self.p'''
            return math.pow(value,1.0/self.p)
          else:
            print('Metric Minkowski error: the length of the variable array ' + str(key) +' is not consistent among the two data sets')
      else:
        print('Metric Minkowski error: the two data sets do not contain the same variables')
    else:
      print('Metric Minkowski error: the structures of the two data sets are different')
