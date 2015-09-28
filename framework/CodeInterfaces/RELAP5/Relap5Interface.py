'''
Created on April 14, 2014

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy
import relapdata
from CodeInterfaceBaseClass import CodeInterfaceBase

class Relap5(CodeInterfaceBase):
  '''this class is used a part of a code dictionary to specialize Model.Code for RELAP5-3D Version 4.0.3'''
  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None):
    '''seek which is which of the input files and generate According the running command'''
    found = False
    for index, inputFile in enumerate(inputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found: raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    outputfile = 'out~'+inputFiles[index].getBase()
    if clargs: addflags = clargs['text']
    else     : addflags = ''
    executeCommand = [('parallel',executable \
                     + ' -i ' + inputFiles[index].getFilename() \
                     + ' -o ' + os.path.join(inputFiles[index].getPath(), inputFiles[index].getBase() + '.o') \
                     + ' -r ' + os.path.join(inputFiles[index].getPath(), inputFiles[index].getBase() + '.r') \
                     + addflags)]
    return executeCommand,outputfile

  def finalizeCodeOutput(self,command,output,workingDir):
    ''' this method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
        It can be used for those codes, that do not create CSV files to convert the whaterver output formato into a csv
        @ command, Input, the command used to run the just ended job
        @ output, Input, the Output name root (string)
        @ workingDir, Input, actual working dir (string)
        @ return is optional, in case the root of the output file gets changed in this method.
    '''
    outfile = os.path.join(workingDir,output+'.o')
    outputobj=relapdata.relapdata(outfile)
    if outputobj.hasAtLeastMinorData(): outputobj.write_csv(os.path.join(workingDir,output+'.csv'))
    else: raise IOError('Relap5 output file '+ command.split('-o')[0].split('-i')[-1].strip()+'.o' + ' does not contain any minor edits. It might be crashed!')

  def checkForOutputFailure(self,output,workingDir):
    """
    this method is called by the RAVEN code at the end of each run  if the return code is == 0.
    This method needs to be implemented by the codes that, if the run fails, return a return code that is 0
    This can happen in those codes that record the failure of the job (e.g. not converged, etc.) as normal termination (returncode == 0)
    This method can be used, for example, to parse the outputfile looking for a special keyword that testifies that a particular job got failed
    (e.g. in RELAP5 would be the keyword "********")
    @ currentInputFiles, currentInputFiles, list,  list of current input files (input files from last this method call)
    @ output, Input, the Output name root (string)
    @ workingDir, Input, actual working dir (string)
    @ return bool, required, True if the job is failed, False otherwise
    """
    from  __builtin__ import any as b_any
    errorWord = "Transient terminated by end of time step cards"
    return not b_any(errorWord in x.strip() for x in open(os.path.join(workingDir,output+'.o'),"r").readlines())

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    '''this generate a new input file depending on which sampler is chosen'''
    import RELAPparser
    self._samplersDictionary                          = {}
    self._samplersDictionary['MonteCarlo'           ] = self.pointSamplerForRELAP5
    self._samplersDictionary['Grid'                 ] = self.pointSamplerForRELAP5
    self._samplersDictionary['Stratified'           ] = self.pointSamplerForRELAP5
    self._samplersDictionary['Adaptive'             ] = self.pointSamplerForRELAP5
    self._samplersDictionary['FactorialDesign'      ] = self.pointSamplerForRELAP5
    self._samplersDictionary['ResponseSurfaceDesign'] = self.pointSamplerForRELAP5
    self._samplersDictionary['DynamicEventTree'     ] = self.DynamicEventTreeForRELAP5
    self._samplersDictionary['BnBDynamicEventTree'  ] = self.DynamicEventTreeForRELAP5
    self._samplersDictionary['StochasticCollocation'] = self.pointSamplerForRELAP5
    found = False
    for index, inputFile in enumerate(currentInputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found: raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    parser = RELAPparser.RELAPparser(currentInputFiles[index].getAbsFile())
    modifDict = self._samplersDictionary[samplerType](**Kwargs)
    parser.modifyOrAdd(modifDict,True)
    newInputFiles = copy.deepcopy(currentInputFiles)
    newInputFiles[index].setBase(Kwargs['prefix']+'~'+currentInputFiles[index].getBase())
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
        #restart_parent = Kwargs['parent_id']+'~restart.r'
        #new_restart = Kwargs['prefix']+'~restart.r'
        #shutil.copyfile(restart_parent,new_restart)
        modifDict['name'] = ['Executioner']
        #modifDict['restart_file_base'] = new_restart
        #print('CODE INTERFACE: Restart file name base is "' + new_restart + '"')
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
