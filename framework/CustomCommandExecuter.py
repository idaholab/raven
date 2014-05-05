'''
Created on April 10, 2014

@author: alfoa
'''

def execCommandReturn(commandString):
  exec('returnedCommand = ' + commandString)
  return returnedCommand
  
def execCommand(commandString):
  exec(commandString)