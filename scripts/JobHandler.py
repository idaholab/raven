'''
Created on Mar 5, 2013

@author: crisr
'''
import Queue as queue
import subprocess
import os
import Datas

class ExternalRunner:
  def __init__(self,command,workingDir,outputFile=None):
    self.command    = command
    self.outputFile = outputFile
    self.workingDir = workingDir
    self.start()
    
  def isDone(self):
    self.process.poll()
    if self.process.returncode != None:
      return True
    else:
      return False
  
  def start(self):
    print(self.command)
    os.chdir(self.workingDir)
    self.process = subprocess.Popen(self.command,stderr=subprocess.STDOUT)
    



class JobHandler:
  def __init__(self):
    self.runInfoDict       = {}
    self.external          = True
    self.mpiCommand        = ''
    self.threadingCommand  = ''
    self.submitDict = {}
    self.submitDict['External'] = self.addExternal
    self.submitDict['Internal'] = self.addInternal
    self.externalRunning        = []
    self.internalRunning        = []
    
  def initialize(self,runInfoDict):
    self.runInfoDict = runInfoDict
    if self.runInfoDict['ParallelProcNumb'] !=1:
      self.mpiCommand = self.runInfoDict['ParallelCommand']+' '+self.runInfoDict['ParallelProcNumb']
    if self.runInfoDict['ThreadingProcessor'] !=1:
      self.threadingCommand = self.runInfoDict['ThreadingCommand'] +' '+self.runInfoDict['ThreadingProcessor']
    self.queue = queue.Queue(self.runInfoDict['batchSize'])
    self.externalRunning        = [None]*self.runInfoDict['batchSize']
    self.internalRunning        = [None]*self.runInfoDict['batchSize']
    #initialize PBS

  def addExternal(self,executeCommand,outputData,outputFile,workingDir):
    #probably something more for the PBS
    command = self.mpiCommand+' '+self.threadingCommand+' '+executeCommand
    print (command)
    return ExternalRunner(command,workingDir,outputFile)

  def addInternal(self):
    return




