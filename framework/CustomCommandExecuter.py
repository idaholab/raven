"""
Created on April 10, 2014

@author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

import copy

def execCommandReturn(commandString,self=None,object=None):
  """
    Method to execute a command, compiled at run time, returning the response
    @ In, commandString, string, the command to execute
    @ In, self, instance, optional, self instance
    @ In, object, instance, optional, object instance
    @ Out, returnedCommand, object, whatever the command needs to return
  """
  exec('returnedCommand = ' + commandString)
  return returnedCommand

def execCommand(commandString,self=None,object=None):
  """
    Method to execute a command, compiled at run time, without returning the response
    @ In, commandString, string, the command to execute
    @ In, self, instance, optional, self instance
    @ In, object, instance, optional, object instance
    @ Out, None
  """
  execCommandReturn(commandString,self,object)
