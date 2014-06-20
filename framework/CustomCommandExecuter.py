'''
Created on April 10, 2014

@author: alfoa
'''

def execCommandReturn(commandString,self=None):
  exec('returnedCommand = ' + commandString)
  return returnedCommand
  
def execCommand(commandString,self=None):
  exec(commandString)