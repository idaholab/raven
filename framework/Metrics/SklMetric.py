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
import utils
import sklearn.metrics.pairwise as pairwise
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Metric import Metric

#Internal Modules End--------------------------------------------------------------------------------

class SKL(Metric):
  """
    Scikit-learn metrics which can be employed only for PointSets
  """
  def initialize(self,inputDict):
    """
      This method initializes the metric object
      @ In, inputDict, dict, dictionary containing initialization parameters
      @ Out, none
    """
    self.metricType = None

  def _localReadMoreXML(self,xmlNode):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initializes internal parameters
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    self.distParams = {}
    for child in xmlNode:
      if child.tag == 'metricType':
        self.metricType = child.text
      else:
        self.distParams[str(child.tag)] = utils.tryParse(child.text)

    availableMetrics = pairwise.kernel_metrics().keys() + pairwise.distance_metrics().keys()
    if self.metricType not in availableMetrics:
      metricList = ', '.join(availableMetrics[:-1]) + ', or ' + availableMetrics[-1]
      self.raiseAnError(IOError,'Metric SKL error: metricType ' + str(self.metricType) + ' is not available. Available metrics are: ' + metricList + '.')

  def distance(self, x, y=None, **kwargs):
    """
      This method returns the distance between two points x and y. If y is not provided then x is a pointSet and a distance matrix is returned
      @ In, x, dict, dictionary containing data of x
      @ In, y, dict, dictionary containing data of y
      @ Out, value, float or numpy.ndarray, distance between x and y (if y is provided) or a square distance matrix if y is None
    """
    if y is not None:
      if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
      # To the reviewer: double check the following line
      # I am not sure if this is correct/optimal. Maybe Josh can have give some directions
        dictTemp = utils.mergeDictionaries(kwargs,self.distParams)
        if self.metricType in pairwise.kernel_metrics().keys():
          value = pairwise.kernel_metrics(X=x, Y=y, metric=self.metricType, **dictTemp)
        elif self.metricType in pairwise.distance_metrics():
          value = pairwise.pairwise_distances(X=x, Y=y, metric=self.metricType, **dictTemp)
        return value
      else:
        self.raiseAnError(IOError,'Metric SKL error: SKL metrics support only PointSets and not HistorySets')
    else:
      if self.metricType == 'mahalanobis':
        covMAtrix = np.cov(x.T)
        kwargs['VI'] = np.linalg.inv(covMAtrix)
      # To the reviewer: double check the following line
      # I am not sure if this is correct/optimal. Maybe Josh can have give some directions
      dictTemp = utils.mergeDictionaries(kwargs,self.distParams)
      if self.metricType in pairwise.kernel_metrics().keys():
        value = pairwise.pairwise_kernels(X=x, metric=self.metricType, **dictTemp)
      elif self.metricType in pairwise.distance_metrics().keys():
        value = pairwise.pairwise_distances(X=x, metric=self.metricType, **dictTemp)
      return value



