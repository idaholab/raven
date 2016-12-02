"""
Created on August 20 2016

@author: mandd
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
import sklearn.metrics.pairwise as pairwise
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

class SKL(Metric):
  """
    Scikit-learn metrics which can be employed only for PointSets
  """
  def initialize(self,inputDict):
    """
      This method initialize the metric object
      @ In, inputDict, dict, dictionary containing initialization parameters
      @ Out, none
    """
    self.metricType = None

  def _localReadMoreXML(self,xmlNode):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initialize internal parameters
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    self.distParams = {}
    for child in xmlNode:
      if child.tag == 'metricType':
        self.metricType = child.text
      else:
        self.distParams[child] = child.text

    
  def distance(self, x, y=None, **kwargs):
    """
      This method set the data return the distance between two points x and y
      @ In, x, dict, dictionary containing data of x
      @ In, y, dict, dictionary containing data of y
      @ Out, value, float, distance between x and y
    """
    if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
      if self.metricType in pairwise.kernel_metrics().keys():
        value = pairwise.kernel_metrics()[self.metricType](X=x, Y=y, dict(**kwargs.items() + self.distParams.items()))
      elif self.metricType in pairwise.distance_metrics():
        value = pairwise.pairwise_distances()[self.metricType](X=x, Y=y, dict(**kwargs.items() + self.distParams.items()))       
      return value
    else:
      self.raiseAnError(IOError,'Metric SKL error: the structures of the two data sets are different')
