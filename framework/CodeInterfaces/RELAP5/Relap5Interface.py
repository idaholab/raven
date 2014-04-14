'''
Created on April 14, 2014

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy

class Relap5Interface:
  '''this class is used a part of a code dictionary to specialize Model.Code for RELAP5-3D Version 4.0.3'''
  def generateCommand(self,inputFiles,executable):
    '''seek which is which of the input files and generate According the running command'''
    if inputFiles[0].endswith('.i'): index = 0
    else: index = 1
    outputfile = 'out~'+os.path.split(inputFiles[index])[1].split('.')[0]
    #   executeCommand will consist of a simple RELAP script that runs relap for inputfile
    #   extracts data and stores in csv file format
    executeCommand = (executable+' '+os.path.split(inputFiles[index])[1]+' ' +
    outputfile)
    return executeCommand,outputfile

  def appendLoadFileExtension(self,fileRoot):
    '''  '''
    return fileRoot + '.csv'

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    '''this generate a new input file depending on which sampler is chosen'''
    import RELAPparser
    self._samplersDictionary                          = {}
    self._samplersDictionary['MonteCarlo'           ] = self.pointSamplerForRELAP5
    self._samplersDictionary['Grid'                 ] = self.pointSamplerForRELAP5
    self._samplersDictionary['LHS'                  ] = self.pointSamplerForRELAP5
    self._samplersDictionary['DynamicEventTree'     ] = self.DynamicEventTreeForRELAP5
    self._samplersDictionary['StochasticCollocation'] = self.pointSamplerForRELAP5
    if currentInputFiles[0].endswith('.i'): index = 0
    else: index = 1
    parser = RELAPparser.RELAPparser(currentInputFiles[index])
    modifDict = self._samplersDictionary[samplerType](**Kwargs)
    parser.modifyOrAdd(modifDict,True)
    temp = str(oriInputFiles[index][:])
    newInputFiles = copy.deepcopy(currentInputFiles)
    newInputFiles[index] = copy.deepcopy(os.path.join(os.path.split(temp)[0],Kwargs['prefix']+"~"+os.path.split(temp)[1]))
    parser.printInput(newInputFiles[index])
    return newInputFiles

  def pointSamplerForRELAP5(self,**Kwargs):
    modifDict = {}
    for keys in Kwargs['SampledVars']:
      key = keys.split(':')
      if len(key) > 1:    Kwargs['SampledVars'][keys]['position'] = int(key[1])
      else: Kwargs['SampledVars'][keys]['position'] = 0
      modifDict[key[0]]=Kwargs['SampledVars'][keys]
    return modifDict

  def DynamicEventTreeForRELAP5(self,**Kwargs):
    raise IOError('DynamicEventTreeForRELAP not yet implemented')
    listDict = []
    return listDict
