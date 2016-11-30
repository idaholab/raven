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
    Scikit-learn  metrics which can be employed only for PointSets
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
    self.requiredKeywords = set(['metricType'])
    self.wrongKeywords = set()
    for child in xmlNode:
      if child.tag == 'metricType':
        self.localDistance = child.text
        self.requiredKeywords.remove('metricType')
      if child.tag not in self.requiredKeywords:
        self.wrongKeywords.add(child.tag)

    if self.requiredKeywords:
      self.raiseAnError(IOError,'The SKL metric is missing the following parameters: ' + str(self.requiredKeywords))
    if not self.wrongKeywords:
      self.raiseAnError(IOError,'The SKL metric block contains parameters that are not recognized: ' + str(self.wrongKeywords))
      
  def distance(self,x):    
    """
      This method set the data return the distance matrix assuming a PointSet is provided 
      @ In, x, dict, dictionary containing data of x
      @ Out, value, float, distance between x and y
    """
    
  def distance(self,x,y):
    """
      This method set the data return the distance between two points x and y
      @ In, x, dict, dictionary containing data of x
      @ In, y, dict, dictionary containing data of y
      @ Out, value, float, distance between x and y
    """
    tempX = copy.deepcopy(x)
    tempY = copy.deepcopy(y)
    if isinstance(tempX,np.ndarray) and isinstance(tempY,np.ndarray):
      self.raiseAnError(IOError,'The DTW metrics is being used only for historySet')
    elif isinstance(tempX,dict) and isinstance(tempY,dict):
      if tempX.keys() == tempY.keys():
        timeLengthX = tempX[self.pivotParameter].size
        timeLengthY = tempY[self.pivotParameter].size
        del tempX[self.pivotParameter]
        del tempY[self.pivotParameter]
        X = np.empty((len(tempX.keys()),timeLengthX))
        Y = np.empty((len(tempY.keys()),timeLengthY))
        for index, key in enumerate(tempX):
          if self.order == 1:
            tempX[key] = np.gradient(tempX[key])
            tempY[key] = np.gradient(tempY[key])
          X[index] = tempX[key]
          Y[index] = tempY[key]
        value = self.dtwDistance(X,Y)
        return value
      else:
        self.raiseAnError('Metric DTW error: the two data sets do not contain the same variables')
    else:
      self.raiseAnError('Metric DTW error: the structures of the two data sets are different')
