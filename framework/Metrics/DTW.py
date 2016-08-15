#External Modules------------------------------------------------------------------------------------
import os
import shutil
import math
import numpy as np
import abc
import importlib
import inspect
import atexit
from sklearn.metrics.pairwise import *
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
    self.allowedKeywords = Set(['pivotParameter','order','localDistance'])
    self.pivotParameter = None
    self.order          = None
    self.localDistance  = None

  def _readMoreXML(self,xmlNode):
    self.wrongKeywords = Set([])
    
    for child in xmlNode:
      if child.tag == 'order':
        if child.text in [0,1]:
          self.order = float(child.text)
        else:
          self.raiseAnError(IOError,'DTW metrics - specified order ' + str(child.text) + ' is not recognized (allowed values are 0 or 1)')
        self.allowedKeywords.pop('order')
      if child.tag == 'pivotParameter':
        self.pivotParameter = child.text
        self.allowedKeywords.pop('pivotParameter')
      if child.tag == 'localDistance':
        self.localDistance = child.text
        self.allowedKeywords.pop('localDistance')
      if child.tag not in self.allowedKeywords:
        self.wrongKeywords.add(child.text)
    
    if not self.allowedKeywords.isEmpty():
      self.raiseAnError(IOError,'The DTW metrics is missing the following parameters: ' + str(self.allowedKeywords))
    if not self.wrongKeywords.isEmpty():
      self.raiseAnError(IOError,'The DTW metrics block contained the following not recognized parameters: ' + str(self.wrongKeywords))
        
  def distance(self,x,y):
    if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
      self.raiseAnError(IOError,'The DTW metrics is being used only for historySet')
    elif isinstance(x,dict) and isinstance(y,dict):
      if x.keys() == y.keys():
        X = np.empty(x.keys().len,x[self.pivotParameter].size)
        Y = np.empty(y.keys().len,y[self.pivotParameter].size)
        index=0
        for key in x.keys():
          if key is not self.pivotParameter:
            X[index] = x[key]
            Y[index] = y[key]
        if self.order == 1.0:
          X,Y = derivative(X,Y)
        value = dtwDistance(X,Y)
      else:
        print('Metric DTW error: the two data sets do not contain the same variables')
    else:
      print('Metric DTW error: the structures of the two data sets are different')
      
  def dtwDistance(self,X,Y):
    tempMatrix = np.zeros((len(X) + 1, len(Y) + 1))
    tempMatrix[0, 1:] = np.inf
    tempMatrix[1:, 0] = np.inf
    distMatrix = tempMatrix[1:, 1:] 
    for i in range(len(x[0])):
        for j in range(len(y[0])):
            distMatrix[i,j] = self.localDistance(X[:][i], Y[:][j])
    C = distMatrix.copy()
    for i in range(len(x)):
        for j in range(len(y)):
            distMatrix[i,j] += min(tempMatrix[i, j], tempMatrix[i, j+1], tempMatrix[i+1, j])
    if len(X)==1:
        path = zeros(len(y)), range(len(y))
    elif len(Y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = tracePath(tempMatrix)
    return distMatrix[-1, -1] / sum(distMatrix.shape)    

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
  
  def derivative(self,X,Y):
    
