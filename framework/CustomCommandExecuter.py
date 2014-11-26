'''
Created on April 10, 2014

@author: alfoa
'''
import copy

def execCommandReturn(commandString,self=None,object=None):
  exec('returnedCommand = ' + commandString)
  return returnedCommand

def execCommand(commandString,self=None,object=None):
  exec(commandString)
