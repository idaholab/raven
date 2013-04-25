'''
Created on Mar 5, 2013

@author: crisr
'''
import Queue as queue
import subprocess
import os
import time
import Datas
import copy

class ExternalRunner:
  def __init__(self,command,workingDir,output=None):
    self.command    = command
    if    output!=None: 
      self.output = output
      self.identifier =  str(output).split("~")[1]
    else: 
      os.path.join(workingDir,'generalOut')
    self.workingDir = workingDir
    self.start()
    
  def isDone(self):
    self.process.poll()
    if self.process.returncode != None:
      
      return True
    else:
      return False
  
  def start(self):
    oldDir = os.getcwd()
    os.chdir(self.workingDir)
    localenv = dict(os.environ)
    localenv['PYTHONPATH'] = ''
    outFile = open(self.output,'w')
    self.process = subprocess.Popen(self.command,shell=True,stdout=outFile,stderr=outFile,cwd=self.workingDir,env=localenv)
    os.chdir(oldDir)
    
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
    #initialize PBS

  def addExternal(self,executeCommand,outputFile,workingDir):
    #probably something more for the PBS
    command = ''
    if self.mpiCommand !='':
      command += self.mpiCommand+' '
    if self.threadingCommand !='':
      command +=self.threadingCommand+' '
    command += executeCommand
    return ExternalRunner(command,workingDir,outputFile)

  def addInternal(self):
    return




