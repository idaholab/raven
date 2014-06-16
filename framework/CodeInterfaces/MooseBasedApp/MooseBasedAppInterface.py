'''
Created on April 14, 2014

@author: alfoa
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
import copy

class MooseBasedAppInterface:
  '''this class is used as part of a code dictionary to specialize Model.Code for RAVEN'''
  def generateCommand(self,inputFiles,executable):
    '''seek which is which of the input files and generate According the running command'''
    if inputFiles[0].endswith('.i'): index = 0
    else: index = 1
    outputfile = 'out~'+os.path.split(inputFiles[index])[1].split('.')[0]
    executeCommand = (executable+' -i '+os.path.split(inputFiles[index])[1] +
                        ' Outputs/file_base='+ outputfile + ' Outputs/exodus=true'+ 
                        ' Outputs/interval=1'+ ' Outputs/output_initial=true' + ' Outputs/csv=true')
                  
    return executeCommand,outputfile

  def appendLoadFileExtension(self,fileRoot):
    '''  '''
    return fileRoot + '.csv'

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    '''this generate a new input file depending on which sampler has been chosen'''
    import MOOSEparser
    self._samplersDictionary                          = {}
    self._samplersDictionary['MonteCarlo'           ] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['Grid'                 ] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['LHS'                  ] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['DynamicEventTree'     ] = self.dynamicEventTreeForMooseBasedApp
    self._samplersDictionary['StochasticCollocation'] = self.pointSamplerForMooseBasedApp
    self._samplersDictionary['Adaptive']              = self.pointSamplerForMooseBasedApp
    if currentInputFiles[0].endswith('.i'): index = 0
    else: index = 1
    parser = MOOSEparser.MOOSEparser(currentInputFiles[index])
    modifDict = self._samplersDictionary[samplerType](**Kwargs)
    modifDict.append(copy.deepcopy({'name':['Outputs'],'special':set(['erase_block'])}))
#     modifDict.append(copy.deepcopy({'name':['Outputs'],'exodus':'true'}))
#     modifDict.append(copy.deepcopy({'name':['Outputs'],'interval':'1'}))
#     modifDict.append(copy.deepcopy({'name':['Outputs'],'output_initial':'true'}))
#     modifDict.append(copy.deepcopy({'name':['Outputs'],'csv':'true'}))
#     modifDict.append(copy.deepcopy({'name':['Outputs'], 'special':set(['erase_block'])}))
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

  def pointSamplerForMooseBasedApp(self,**Kwargs):
    listDict  = []
    modifDict = {}
    # the position in, eventually, a vector variable is not available yet...
    # the MOOSEparser needs to be modified in order to accept this variable type
    # for now the position (i.e. ':' at the end of a variable name) is discarded
    for var in Kwargs['SampledVars']:
        if 'alias' in Kwargs.keys():
          # for understending the alias system, plase check module Models.py (class Code)
          if var in Kwargs['alias'].keys():
            key = Kwargs['alias'][var].split(':')
            varname = var
        else:
          key = var.split(':')
          varname = key[0]
        modifDict = {}
        modifDict['name'] = []
        modifDict['name'] = key[0].split('|')[:-1]
        modifDict[key[0].split('|')[-1]] = Kwargs['SampledVars'][var]
        listDict.append(modifDict)
        del modifDict
        if 'raven' not in Kwargs['executable'].lower():
          listDict.append({'name':['AuxVariables',varname],'family':'SCALAR'})
          listDict.append({'name':['AuxVariables',varname],'initial_condition':Kwargs['SampledVars'][var]})
          listDict.append({'name':['Postprocessors',varname],'type':'ScalarVariable'})
          listDict.append({'name':['Postprocessors',varname],'variable':varname})
    return listDict

  def dynamicEventTreeForMooseBasedApp(self,**Kwargs):
    raise IOError('dynamicEventTreeForMooseBasedApp not yet implemented')
    listDict = []
    return listDict