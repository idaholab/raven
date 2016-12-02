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
import copy
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
import SupervisedLearning
import PostProcessors #import returnFilterInterface
import CustomCommandExecuter
import utils
import mathUtils
import TreeStructure
import Files

#Internal Modules End--------------------------------------------------------------------------------

class Metric(utils.metaclass_insert(abc.ABCMeta,BaseType)):
  """
    This is the general interface to any RAVEN metric object.
    It contains an initialize, a _readMoreXML, and an evaluation (i.e., distance) methods
  """
  def __init__(self):
    """
      This is the basic method initialize the metric object
      @ In, none
      @ Out, none
    """
    BaseType.__init__(self)
    self.type = self.__class__.__name__
    self.name = self.__class__.__name__

  def initialize(self,inputDict):
    """
      This method initialize each metric object
      @ In, inputDict, dict, dictionary containing initialization parameters
      @ Out, none
    """
    pass

  def _readMoreXML(self,xmlNode):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initialize internal parameters
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    self._localReadMoreXML(xmlNode)


  def distance(self,x,y,**kwargs):
    """
      This method actually calculates the distance between two dataObjects x and y
      @ In, x, dict, dictionary containing data of x
      @ In, y, dict, dictionary containing data of y
      @ In, weights, dictionary of probably weights
      @ Out, value, float, distance between x and y
    """
    pass


