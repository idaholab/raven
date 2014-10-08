'''
Created on April 14, 2014

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy

class Relap5:
  '''this class is used a part of a code dictionary to specialize Model.Code for RELAP5-3D Version 4.0.3'''
  def generateCommand(self,inputFiles,executable,flags=None):
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
    self._samplersDictionary['Adaptive'             ] = self.pointSamplerForRELAP5
    self._samplersDictionary['DynamicEventTree'     ] = self.DynamicEventTreeForRELAP5
    self._samplersDictionary['BnBDynamicEventTree'  ] = self.DynamicEventTreeForRELAP5
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
    listDict = []
    modifDict = {}
    cardList = {}
    for keys in Kwargs['SampledVars']:
      key = keys.split(':')
      if len(key) > 1:
        position=int(key[1])    
        cardList[key[0]]={'position':position,'value':Kwargs['SampledVars'][keys]}
      else: cardList[key[0]]={'position':0,'value':Kwargs['SampledVars'][keys]}
    modifDict['cards']=cardList
    listDict.append(modifDict)
    return listDict

  def DynamicEventTreeForRELAP5(self,**Kwargs):
    listDict =[] 
    cardList={}   #  List of cards to be modified in RELAP5 Input File
    # Check the initiator distributions and add the next threshold
    if 'initiator_distribution' in Kwargs.keys():
      for i in range(len(Kwargs['initiator_distribution'])):
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
      #if Kwargs['end_ts'] != 0 or Kwargs['end_ts'] == 0:

      if str(Kwargs['start_time']) != 'Initial':
        modifDict = {}
#        restart_parent = Kwargs['parent_id']+'~restart.r'
#        new_restart = Kwargs['prefix']+'~restart.r'
#        shutil.copyfile(restart_parent,new_restart)
        modifDict['name'] = ['Executioner']
#        modifDict['restart_file_base'] = new_restart 
#        print('CODE INTERFACE: Restart file name base is "' + new_restart + '"')
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
      if Kwargs['branch_changed_param'][0] not in ('None',b'None'): 
        for i in range(len(Kwargs['branch_changed_param'])):
          modifDict = {}
          modifDict['name'] = ['RestartInitialize',Kwargs['branch_changed_param'][i]]
          modifDict['value'] = Kwargs['branch_changed_param_value'][i]
          listDict.append(modifDict)
          del modifDict
    modifDict={}
    for keys in Kwargs['SampledVars']:
      key = keys.split(':')
      if len(key) > 1:  
        if Kwargs['start_time'] != 'Initial':  cardList[key[0]]={'position':key[1],'value':float(Kwargs['SampledVars'][keys])}
        else: cardList[key[0]]={'position':key[1],'value':Kwargs['SampledVars'][keys]}
      else: 
        if Kwargs['start_time'] != 'Initial':  cardList[key[0]]={'position':0,'value':float(Kwargs['SampledVars'][keys])}
        else: cardList[key[0]]={'position':0,'value':float(Kwargs['SampledVars'][keys])}
    modifDict['cards']=cardList
    if 'aux_vars' in Kwargs.keys():
      for keys in Kwargs['aux_vars']:
        key = keys.split(':')
        if len(key) > 1:  cardList[key[0]]={'position':key[1],'value':Kwargs['aux_vars'][keys]}
        else:  cardList[key[0]]={'position':0,'value':Kwargs['aux_vars'][keys]}
        modifDict['cards']=cardList
    listDict.append(modifDict)
    del modifDict
    return listDict  
