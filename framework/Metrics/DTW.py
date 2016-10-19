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

class DTW(Metric):
  """
    Dynamic Time Warping metrics which can be employed only for historySets
  """
  def initialize(self,inputDict):
    """
      This method initialize the metric object
      @ In, inputDict, dict, dictionary containing initialization parameters
      @ Out, none
    """
    self.pivotParameter = None
    self.order          = None
    self.localDistance  = None

  def readMoreXML(self,xmlNode):
    """
      Method that reads the portion of the xml input that belongs to this specialized class
      and initialize internal parameters
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    self.requiredKeywords = set(['pivotParameter','order','localDistance'])
    self.wrongKeywords = set()
    for child in xmlNode:
      if child.tag == 'order':
        if child.text in ['0','1']:
          self.order = float(child.text)
        else:
          self.raiseAnError(IOError,'DTW metrics - specified order ' + str(child.text) + ' is not recognized (allowed values are 0 or 1)')
        self.requiredKeywords.remove('order')
      if child.tag == 'pivotParameter':
        self.pivotParameter = child.text
        self.requiredKeywords.remove('pivotParameter')
      if child.tag == 'localDistance':
        self.localDistance = child.text
        self.requiredKeywords.remove('localDistance')
      if child.tag not in self.requiredKeywords:
        self.wrongKeywords.add(child.tag)

    if self.requiredKeywords:
      self.raiseAnError(IOError,'The DTW metrics is missing the following parameters: ' + str(self.requiredKeywords))
    if not self.wrongKeywords:
      self.raiseAnError(IOError,'The DTW metrics block contains parameters that are not recognized: ' + str(self.wrongKeywords))

  def distance(self,x,y):
    """
      This method set the data return the distance between two histories x and y
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
        if self.order == 1:
          tempX = self.derivative(tempX)
          tempY = self.derivative(tempY)
        X = np.empty((len(tempX.keys()),timeLengthX))
        Y = np.empty((len(tempY.keys()),timeLengthY))
        for index, key in enumerate(tempX):
          X[index] = tempX[key]
          Y[index] = tempY[key]
        value = self.dtwDistance(X,Y)
        return value
      else:
        self.raiseAnError('Metric DTW error: the two data sets do not contain the same variables')
    else:
      self.raiseAnError('Metric DTW error: the structures of the two data sets are different')

  def dtwDistance(self,x,y):
    """
      This method actually calculates the distance between two histories x and y
      @ In, x, numpy.ndarray, data matrix for x
      @ In, y, numpy.ndarray, data matrix for y
      @ Out, value, float, distance between x and y
    """
    assert len(x)
    assert len(y)
    r, c = len(x[0,:]), len(y[0,:])
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]
    D1 = pairwise.pairwise_distances(x.T,y.T, metric=self.localDistance)
    C = D1.copy()
    for i in range(r):
      for j in range(c):
        D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
      path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
      path = range(len(x)), np.zeros(len(x))
    else:
      path = self.tracePath(D0)
    return D1[-1, -1]

  def tracePath(self,D):
    """
      This method calculate the time warping path given a local distance matrix D
      @ In, D,  numpy.ndarray (2D), local distance matrix D
      @ Out, p, numpy.ndarray (1D), path along horizontal direction
      @ Out, q, numpy.ndarray (1D), path along vertical direction
    """
    i,j = np.array(D.shape) - 2
    p,q = [i], [j]
    while ((i > 0) or (j > 0)):
      tb = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
      if (tb == 0):
          i -= 1
          j -= 1
      elif (tb == 1):
          i -= 1
      else:
          j -= 1
      p.insert(0, i)
      q.insert(0, j)
    return np.array(p), np.array(q)

  def derivative(self,x):
    """
      This method calculate the derivative of an history x. Note that method is computational expensive
      @ In,  x, dict, dictionary containing data of y
      @ Out, y, float, distance between x and y
    """
    y={}
    for key in x.keys():
      y[key] = np.zeros(len(x[key]))
      for i in range(len(x[key])):
        if i==0:
          y[key][i] = 0.0
        elif i == len(x[key])-1:
          y[key][i] = 0.0
        else:
          y[key][i] = ((x[key][i]-x[key][i-1])+(x[key][i+1]-x[key][i-1])/2.0)/2.0
    return y

