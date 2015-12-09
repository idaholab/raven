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
  
  def __init__(self):
    self.testID = None
  
  def run(self,Input):
    print('self.testID: ' + str(self.testID))
    return Input
  
  def finalizeOutput(self,output):
    return output
  
  def _readMoreXML(self,xmlNode):
    for child in xmlNode:
      if child == 'testID':
        self.testID = child.text