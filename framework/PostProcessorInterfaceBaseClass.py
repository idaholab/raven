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
  
  def __init__(self):
    pass

  def readMoreXML(self,xmlNode):
    self._readMoreXML(xmlNode)
    
  def _readMoreXML(self,xmlNode):
    pass
    
  def run(self):
    return
  
  def finalizeOutput(self,output):
    return