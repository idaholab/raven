"""
Created on October 28, 2015

"""

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase

import os
import numpy as np
from scipy import interpolate
import copy

import HistorySetSync

class HistorySetSampling(PostProcessorInterfaceBase):

  def initialize(self):
    """
     Method to initialize the Interfaced Post-processor
     @ In, None,
     @ Out, None,

    """

    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'HistorySet'
    self.outputFormat = 'PointSet'
    
    self.HSsyncPP = HistorySetSync()
    self.HSsyncPP.initialize()

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """
    
    self.HSsyncPP.readMoreXML(xmlNode) 

  def run(self,inputDic, timeInstant):
    """
     Method to post-process the dataObjects
     @ In,  inputDic , dictionary
     @ Out, outputDic, dictionary
    """
    outputDic = {}
    outputHSDic = {}
    
    outputPSDic['metadata'] = copy.deepcopy(inputDic['metadata'])
    outputPSDic['data'] = {}
    outputPSDic['data']['input'] = copy.deepcopy(inputDic['data']['input'])
    outputPSDic['data']['output'] = {}
    
    outputHSDic = self.HSsyncPP.run(inputDic)
    
    outputHSDic = self.timeSnapShot(outputHSDic,timeInstant)

    return outputPSDic
  
  def timeSnapShot(self, inputDic, timeInst):

