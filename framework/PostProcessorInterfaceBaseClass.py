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
  
  def initialize(self):
    self.inputFormat  = None
    self.outputFormat = None

  def readMoreXML(self,xmlNode):
    pass
    
  def run(self,inputDic):
    pass
  
  def checkGeneratedDicts(self,outputDic):
    if self.checkOutputFormat(outputDic['data']['input']) and self.checkOutputFormat(outputDic['data']['output']):
      return True
    else:
      return False
  
  def checkOutputFormat(self,outputDic):
    """ This function check that the generated output dictionary is built accordingly to outputFormat
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
    """ This function check that the generated input dictionary is built accordingly to outputFormat
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