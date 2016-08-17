#External Modules------------------------------------------------------------------------------------
import os
import shutil
import math
import numpy as np
import abc
import importlib
import inspect
import atexit
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

  def initialize(self,inputDict):
    self.pivotParameter = None
    self.order          = None
    self.localDistance  = None

  def _readMoreXML(self,xmlNode):
    self.allowedKeywords = set(['pivotParameter','order','localDistance'])
    self.wrongKeywords = set()
    for child in xmlNode:
      if child.tag == 'order':
        if child.text in ['0','1']:
          self.order = float(child.text)
        else:
          self.raiseAnError(IOError,'DTW metrics - specified order ' + str(child.text) + ' is not recognized (allowed values are 0 or 1)')
        self.allowedKeywords.remove('order')
      if child.tag == 'pivotParameter':
        self.pivotParameter = child.text
        self.allowedKeywords.remove('pivotParameter')
      if child.tag == 'localDistance':
        self.localDistance = child.text
        self.allowedKeywords.remove('localDistance')
      if child.tag not in self.allowedKeywords:
        self.wrongKeywords.add(child.tag)
    
    if self.allowedKeywords:
      self.raiseAnError(IOError,'The DTW metrics is missing the following parameters: ' + str(self.allowedKeywords))
    if not self.wrongKeywords:
      self.raiseAnError(IOError,'The DTW metrics block contains parameters that are not recognized: ' + str(self.wrongKeywords))
        
  def distance(self,x,y):
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
          X[index] = tempX[key]
          Y[index] = tempY[key]
        if self.order == 1:
          X,Y = derivative(X,Y)
        value = self.dtwDistance(X,Y)
        return value 
      else:
        self.raiseAnError('Metric DTW error: the two data sets do not contain the same variables')
    else:
      self.raiseAnError('Metric DTW error: the structures of the two data sets are different')
      
  def dtwDistance(self,x,y):
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:] 
    for i in range(r):
        for j in range(c):
            D1[i, j] = pairwise.pairwise_distances(x[:,i], y[:,j], metric=self.localDistance)
    #D1[i, j] = (pairwise.pairwise_distances(x[:,i], y[:,j], metric=self.localDistance) for i in range(r) for j in range(c))
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
  
#  def derivative(self,X,Y):
    
