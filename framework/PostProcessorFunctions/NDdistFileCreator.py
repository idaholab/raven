
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase


import os
import numpy as np
from scipy import interpolate
import copy


class NDdistFileCreator(PostProcessorInterfaceBase):
  """
  """
  def initialize(self):
    """
     Method to initialize the Interfaced Post-processor
     @ In, None,
     @ Out, None,

    """

    PostProcessorInterfaceBase.initialize(self)
    self.inputFormat  = 'HistorySet'
    self.outputFormat = 'HistorySet'


  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, ElementTree, Xml element node
      @ Out, None
    """

    for child in xmlNode:
      if child.tag == 'fileName':
        self.fileName = child.text
      if child.tag == 'densityEstimation':
        self.densityEstimation = child.text

  def run(self,inputDic):
    return outputDic