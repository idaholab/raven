"""
Created on December 1st, 2015

"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

#External Modules------------------------------------------------------------------------------------
import abc
import os
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
#Internal Modules End--------------------------------------------------------------------------------

class PostProcessorInterfaceBase(utils.metaclass_insert(abc.ABCMeta,object)):
  
  def initialize(self):
    self.inputFormat  = None
    self.outputFormat = None

  def readMoreXML(self,xmlNode):
    pass
    
  def run(self):
    pass
  
  def checkGeneratedDicts(self,inputDic,outputDic):
    if checkInputFormat(inputDic) and checkOutputFormat(outputDic):
      return True
  
  def checkOutputFormat(self,outputDic):
    """ This function check that the generated output dictionary is built accordingly to outputFormat
    """
    outcome = True    
    if outputDic is dict:
      if self.outputFormat == 'HistorySet' or self.outputFormat == 'History': 
        for key in outputDic:
          if type(key.value) is dict:
            outcome = outcome and True
          else:
            outcome = outcome and False
          for keys in key:
            if isinstance(keys.value,np.ndarray):
              outcome = outcome and True
            else:
              outcome = outcome and False
        if outcome==False:
          self.raiseAnError("Interfaced Post-Processor: output type is not consistent with ")
      else: # self.outputFormat == 'PointSet or self.outputFormat == 'PointSet':
        for key in outputDic:
          if isinstance(key.value,np.ndarray):
            outcome = outcome and True
          else:
            outcome = outcome and False
    else:
      outcome = False
    
    if outcome==False:
      self.raiseAnError("Interfaced Post-Processor: output type is not a dictionary")
    
    return outcome
  
  def checkInputFormat(self,inputDic):
    """ This function check that the generated input dictionary is built accordingly to outputFormat
    """
    outcome = True    
    if inputDic is dict: 
      for key in inputDic:
        if isinstance(key.value,np.ndarray):
          outcome = outcome and True
        else:
          outcome = outcome and False         
    else:
      outcome = False
    if outcome==False:
      self.raiseAnError("Interfaced Post-Processor: output type is not a dictionary")