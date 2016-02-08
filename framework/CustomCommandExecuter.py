"""
Created on April 10, 2014

@author: alfoa
"""
import copy

def execCommandReturn(commandString,self=None,object=None):
  exec('returnedCommand = ' + commandString)
  return returnedCommand

def execCommand(commandString,self=None,object=None):
  print '\n\n\nDEBUG:\n',commandString
  print 'self:',self
  print 'object:',object
  exec(commandString)
