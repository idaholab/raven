'''
Created on December 1, 2015

'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase
import numpy as np

class testInterfacedPP(PostProcessorInterfaceBase):
  
  def initialize(self):
    self.inputFormat  = 'HistorySet'
    self.outputFormat = 'HistorySet'
  
  def run(self,inInputDic,inOutputDic):   
    return inInputDic,inOutputDic
  
  def readMoreXML(self,xmlNode):
    for child in xmlNode:
      if child.tag == 'testID':
        self.testID = child.text 