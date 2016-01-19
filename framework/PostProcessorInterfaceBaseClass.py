"""
Created on December 1st, 2015

"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

#External Modules------------------------------------------------------------------------------------
import abc
import os
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from cached_ndarray import c1darray
import utils
#Internal Modules End--------------------------------------------------------------------------------

class PostProcessorInterfaceBase(utils.metaclass_insert(abc.ABCMeta,object)):
  """ This class is the base interfaced post-processor clas
      It contains the three methods that need to be implemented:
      - initialize
      - run
      - readMoreXML
  """
  def initialize(self):
    """
     Method to initialize the Interfaced Post-processor
     @ In, None,
     @ In, None,

    """
    self.inputFormat  = None
    self.outputFormat = None

  def readMoreXML(self,xmlNode):
    """
      Function that reads elements this post-processor will use
      @ In, xmlNode, Xml element node
      @ Out, None
    """
    pass

  def run(self,inputDic):
    """
     Method to post-process the dataObjects
     @ In, inputDic, dictionary
     @ Out, dictionary

    """
    pass

  def checkGeneratedDicts(self,outputDic):
    """
     Method to check that dictionary generated in def run(self, inputDic) is consistent
     @ In, outputDic, dictionary
     @ Out, boolean, 
    """
    if self.checkOutputFormat(outputDic['data']['input']) and self.checkOutputFormat(outputDic['data']['output']):
      return True
    else:
      return False

  def checkOutputFormat(self,outputDic):
    """
     This method checks that the generated output dictionary is built accordingly to outputFormat
     @ In, outputDic, dictionary
     @ Out, boolean, outcome
    """

    outcome = True
    if isinstance(outputDic,dict):
      if self.outputFormat == 'HistorySet' or self.outputFormat == 'History':
        for key in outputDic:
          if isinstance(outputDic[key],dict):
            outcome = outcome and True
          else:
            outcome = outcome and False
          for keys in outputDic[key]:
            if isinstance(outputDic[key][keys],(np.ndarray,c1darray)):
              outcome = outcome and True
            else:
              outcome = outcome and False
      else:
        for key in outputDic:
          if isinstance(outputDic[key],np.ndarray):
            outcome = outcome and True
          else:
            outcome = outcome and False
    else:
      outcome = outcome and False
    return outcome

  def checkInputFormat(self,inputDic):
    """
     This method checks that the generated input dictionary is built accordingly to outputFormat
     @ In, outputDic, dictionary
     @ Out, boolean, outcome
    """
    
    outcome = True
    if isinstance(inputDic,dict):
      for key in inputDic:
        if isinstance(key.value,np.ndarray):
          outcome = outcome and True
        else:
          outcome = outcome and False
    else:
      outcome = False
    return outcome
