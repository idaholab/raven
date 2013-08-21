'''
Created on Jun 8, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
import shutil
import Datas
import numpy as np
import ast
from BaseType import BaseType

class RavenInterface:
  '''this class is used as part of a code dictionary to specialize Model.Code for RAVEN'''
  def generateCommand(self,inputFiles,executable):
    '''seek which is which of the input files and generate According the running command'''
    if inputFiles[0].endswith('.i'): index = 0
    else: index = 1
    outputfile = 'outFrom~'+os.path.split(inputFiles[index])[1].split('.')[0]
    executeCommand = (executable+' -i '+os.path.split(inputFiles[index])[1]+' Output/postprocessor_csv=true' + 
    ' Output/file_base='+ outputfile)
    return executeCommand,outputfile

  def appendLoadFileExtension(self,fileRoot):
    '''  '''
    return fileRoot + '.csv'

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    '''this generate a new input file depending on which sampler has been chosen'''
    import MOOSEparser
    newInputFiles = []
    self.samplersDictionary                          = {}
    self.samplersDictionary['MonteCarlo']            = self.MonteCarloForRAVEN
    self.samplersDictionary['EquallySpaced']         = self.EquallySpacedForRAVEN
    self.samplersDictionary['LatinHyperCube']        = self.LatinHyperCubeForRAVEN
    self.samplersDictionary['DynamicEventTree']      = self.DynamicEventTreeForRAVEN
    self.samplersDictionary['StochasticCollocation'] = self.StochasticCollocationForRAVEN
    if currentInputFiles[0].endswith('.i'): index = 0
    else: index = 1
    parser = MOOSEparser.MOOSEparser(currentInputFiles[index])
    modifDict = self.samplersDictionary[samplerType](**Kwargs)
    parser.modifyOrAdd(modifDict,False)
    temp = str(oriInputFiles[index][:])
    newInputFiles = copy.deepcopy(currentInputFiles)
    #TODO fix this? storing unwieldy amounts of data in 'prefix'
    if type(Kwargs['prefix']) in [str,type("")]:#Specifing string type for python 2 and 3
      newInputFiles[index] = copy.deepcopy(os.path.join(os.path.split(temp)[0],Kwargs['prefix']+"~"+os.path.split(temp)[1]))
    else:
      newInputFiles[index] = copy.deepcopy(os.path.join(os.path.split(temp)[0],str(Kwargs['prefix'][1][0])+"~"+os.path.split(temp)[1]))
    parser.printInput(newInputFiles[index])
    return newInputFiles

  def StochasticCollocationForRAVEN(self,**Kwargs):
    try: counter = Kwargs['prefix']
    except: raise IOError('a counter is (currently) needed for the StochColl sampler for RAVEN')
    #try: qps = Kwargs['qps']
    #except: raise IOError('a qp index is required for the StochColl sampler for RAVEN')
    listDict = []
    varValDict = Kwargs['vars'] #come in as a string of a list, need to re-list
    #print('\nvarValDict type:',type(varValDict),varValDict,'\n')
    #qps   = Kwargs['qps']
    #names = ast.literal_eval(varValDict.keys()) #turns string of list/tuple into list
    #vals = ast.literal_eval(var
    #qps   = ast.literal_eval(qps)
    for key in varValDict.keys():
      modifDict={}
      modifDict['name']=key.split(':')
      modifDict['value']=varValDict[key]
      #print('interface: set',key.split(':'),'to',varValDict[key])
      listDict.append(modifDict)
      del modifDict
    return listDict

  def MonteCarloForRAVEN(self,**Kwargs):
    try: counter = Kwargs['prefix']
    except: raise IOError('a counter is needed for the Monte Carlo sampler for RAVEN')
    try: init_seed = Kwargs['initial_seed']
    except: init_seed = 1
    
    listDict = []
    modifDict = {}
    modifDict['name'] = ['Distributions']
    RNG_seed = int(counter) + int(init_seed) - 1
    modifDict['RNG_seed'] = str(RNG_seed)
    listDict.append(modifDict)
    return listDict
  
  def DynamicEventTreeForRAVEN(self,**Kwargs):
    listDict = []
    # Check the initiator distributions and add the next threshold
    if 'initiator_distribution' in Kwargs.keys():
      for i in xrange(len(Kwargs['initiator_distribution'])):
        modifDict = {}
        modifDict['name'] = ['Distributions',Kwargs['initiator_distribution'][i]]
        modifDict['ProbabilityThreshold'] = Kwargs['PbThreshold'][i]
        listDict.append(modifDict)
        del modifDict
    # add the initial time for this new branch calculation
    if 'start_time' in Kwargs.keys():
      if Kwargs['start_time'] != 'Initial':
        modifDict = {}
        st_time = Kwargs['start_time']
        modifDict['name'] = ['Executioner']
        modifDict['start_time'] = st_time
        listDict.append(modifDict)
        del modifDict
    # create the restart file name root from the parent branch calculation
    # in order to restart the calc from the last point in time
    if 'end_ts' in Kwargs.keys():
      if Kwargs['end_ts'] != 0:
        modifDict = {}
        end_ts_str = str(Kwargs['end_ts'])
        if(Kwargs['end_ts'] <= 9999):
          n_zeros = 4 - len(end_ts_str)
          for i in xrange(n_zeros):
            end_ts_str = "0" + end_ts_str
        
        splitted = Kwargs['outfile'].split('~')
        output_parent = splitted[0] + '~' + Kwargs['parent_id'] + '~' + splitted[1]
        restart_file_base = output_parent + "_restart_" + end_ts_str      
        modifDict['name'] = ['Executioner']
        modifDict['restart_file_base'] = restart_file_base
        listDict.append(modifDict)
        del modifDict
    # max simulation time (if present)
    if 'end_time' in Kwargs.keys():
      modifDict = {}
      end_time = Kwargs['end_time']
      modifDict['name'] = ['Executioner']
      modifDict['end_time'] = end_time
      listDict.append(modifDict)
      del modifDict
      
    modifDict = {}
    modifDict['name'] = ['Output']
    modifDict['num_restart_files'] = 1
    listDict.append(modifDict)
    del modifDict
    # in this way we erase the whole block in order to neglect eventual older info
    # remember this "command" must be added before giving the info for refilling the block
    modifDict = {}
    modifDict['name'] = ['RestartInitialize']
    modifDict['erase_block'] = True
    listDict.append(modifDict)
    
    del modifDict    
    # check and add the variables that have been changed by a distribution trigger
    # add them into the RestartInitialize block
    if 'branch_changed_param' in Kwargs.keys():
      if Kwargs['branch_changed_param'][0] != 'None': 
        for i in xrange(len(Kwargs['branch_changed_param'])):
          modifDict = {}
          modifDict['name'] = ['RestartInitialize',Kwargs['branch_changed_param'][i]]
          modifDict['value'] = Kwargs['branch_changed_param_value'][i]
          listDict.append(modifDict)
          del modifDict
    return listDict  

  def EquallySpacedForRAVEN(self,**Kwargs):
    raise IOError('EquallySpacedForRAVEN not yet implemented')
    listDict = []
    return listDict
  
  def LatinHyperCubeForRAVEN(self,**Kwargs):
    raise IOError('LatinHyperCubeForRAVEN not yet implemented')
    listDict = []
    return listDict
  
class RelapInterface:
  '''this class is used a part of a code dictionary to specialize Model.Code for RELAP5-3D Version 4.0.3'''
  def generateCommand(self,inputFiles,executable):
    '''seek which is which of the input files and generate According the running command'''
    if inputFiles[0].endswith('.i'): index = 0
    else: index = 1
    outputfile = 'outFrom~'+os.path.split(inputFiles[index])[1].split('.')[0]
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
    newInputFiles = []
    self.samplersDictionary                     = {}
    self.samplersDictionary['MonteCarlo']       = self.MonteCarloForRELAP
    self.samplersDictionary['EquallySpaced']    = self.EquallySpacedForRELAP
    self.samplersDictionary['LatinHyperCube']   = self.LatinHyperCubeForRELAP
    self.samplersDictionary['DynamicEventTree'] = self.DynamicEventTreeForRELAP
    if currentInputFiles[0].endswith('.i'): index = 0
    else: index = 1
    parser = RELAPparser.RELAPparser(currentInputFiles[index])
    modifDict = self.samplersDictionary[samplerType](**Kwargs)
    parser.modifyOrAdd(modifDict,True)
    temp = str(oriInputFiles[index][:])
    newInputFiles = copy.deepcopy(currentInputFiles)
    newInputFiles[index] = copy.deepcopy(os.path.join(os.path.split(temp)[0],Kwargs['prefix']+"~"+os.path.split(temp)[1]))
    parser.printInput(newInputFiles[index])
    return newInputFiles
    
  def MonteCarloForRELAP(self,**Kwargs):
    try: counter = Kwargs['prefix']
    except: raise IOError('a counter is needed for the Monte Carlo sampler for RAVEN')
    listDict = []
    modifDict = {}
    for keys in Kwargs:
      if 'position' in Kwargs[keys]:
        modifDict[keys]=Kwargs[keys]
    return modifDict
    
  def DynamicEventTreeForRELAP(self,**Kwargs):
    raise IOError('DynamicEventTreeForRELAP not yet implemented')
    listDict = []
    return listDict

  def EquallySpacedForRELAP(self,**Kwargs):
    raise IOError('EquallySpacedForRAVEN not yet implemented')
    listDict = []
    return listDict
  
  def LatinHyperCubeForRELAP(self,**Kwargs):
    raise IOError('LatinHyperCubeForRAVEN not yet implemented')
    listDict = []
    return listDict

  
class ExternalTest:
  def generateCommand(self,inputFiles,executable):
    return '', ''

  def findOutputFile(self,command):
    return ''
  
def returnCodeInterface(Type):
  '''this allow to the code(model) class to interact with a specific
     code for which the interface is present in the CodeInterfaces module'''
  base = 'Code'
  codeInterfaceDict = {}
  codeInterfaceDict['RAVEN'] = RavenInterface
  codeInterfaceDict['ExternalTest'] = ExternalTest
  codeInterfaceDict['RELAP5'] = RelapInterface
  try: return codeInterfaceDict[Type]()
  except: raise NameError('not known '+base+' type '+Type)

