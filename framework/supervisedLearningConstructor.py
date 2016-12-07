"""
Created on December 6, 2016

@author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------

#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
import utils
#Internal Modules End--------------------------------------------------------------------------------
class supervisedLearningConstructor(BaseType):
  """
    This class represents an interface with all the supervised learning algorithms 
    It is a utility class needed to hide the discernment between time-dependent and static 
    surrogate models
  """
  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately initialize a supervised learning object (static or time-dependent)
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    pass