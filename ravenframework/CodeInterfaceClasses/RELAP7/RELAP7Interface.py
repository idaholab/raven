# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on April 14, 2014

@author: alfoa
"""
import os
import sys
import copy
from ravenframework.utils import utils
import json
uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from ..MooseBasedApp import MOOSEparser
from ..MooseBasedApp import MooseInputParser

class RELAP7(CodeInterfaceBase):
  """
    This class is used as part of a code dictionary to specialize Model.Code for RELAP7
  """
  def generateCommand(self,inputFiles,executable,clargs=None,fargs=None,preExec=None):
    """
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the axuiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ In, preExec, string, optional, a string the command that needs to be pre-executed before the actual command here defined
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
    """
    found = False
    for index, inputFile in enumerate(inputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found:
      raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    outputfile = 'out~'+inputFiles[index].getBase()

    if clargs:
      precommand = executable + clargs['text']
    else:
      precommand = executable

    if executable.strip().endswith("py"):
      # for testing
      precommand = "python "+ precommand

    executeCommand = [('parallel',precommand + ' -i '+inputFiles[index].getFilename() + ' Outputs/file_base='+ outputfile +
                      ' Outputs/csv=false' + ' Outputs/checkpoint=true'+ ' Outputs/tail/type=ControlLogicBranchingInfo'+
                      ' Outputs/ravenCSV/type=CSVRaven')]
    returnCommand = executeCommand,outputfile
    return returnCommand

  def createNewInput(self,currentInputFiles,oriInputFiles,samplerType,**Kwargs):
    """
      This method is used to generate an input based on the information passed in.
      @ In, currentInputFiles, list,  list of current input files (input files from last this method call)
      @ In, oriInputFiles, list, list of the original input files
      @ In, samplerType, string, Sampler type (e.g. MonteCarlo, Adaptive, etc. see manual Samplers section)
      @ In, Kwargs, dictionary, kwarded dictionary of parameters. In this dictionary there is another dictionary called "SampledVars"
             where RELAP7 stores the variables that got sampled (e.g. Kwargs['SampledVars'] => {'var1':10,'var2':40})
      @ Out, newInputFiles, list, list of newer input files, list of the new input files (modified and not)
    """
    self._samplersDictionary                             = {}
    self._samplersDictionary[samplerType]                = self.gridForRELAP7
    self._samplersDictionary['MonteCarlo'              ] = self.monteCarloForRELAP7
    self._samplersDictionary['Grid'                    ] = self.gridForRELAP7
    self._samplersDictionary['LimitSurfaceSearch'      ] = self.gridForRELAP7 # same Grid Fashion. It forces a dist to give a particular value
    self._samplersDictionary['Stratified'              ] = self.latinHyperCubeForRELAP7
    self._samplersDictionary['DynamicEventTree'        ] = self.dynamicEventTreeForRELAP7
    self._samplersDictionary['FactorialDesign'         ] = self.gridForRELAP7
    self._samplersDictionary['ResponseSurfaceDesign'   ] = self.gridForRELAP7
    self._samplersDictionary['AdaptiveDynamicEventTree'] = self.dynamicEventTreeForRELAP7
    self._samplersDictionary['StochasticCollocation'   ] = self.gridForRELAP7
    self._samplersDictionary['CustomSampler'           ] = self.gridForRELAP7

    found = False
    for index, inputFile in enumerate(currentInputFiles):
      if inputFile.getExt() in self.getInputExtension():
        found = True
        break
    if not found:
      raise IOError('None of the input files has one of the following extensions: ' + ' '.join(self.getInputExtension()))
    parser = MOOSEparser.MOOSEparser(currentInputFiles[index].getAbsFile())
    Kwargs["distributionNode"] = MooseInputParser.findInGetpot(parser.roots, ["Distributions"])
    # OLD Kwargs["distributionNode"] = parser.findNodeInXML("Distributions")
    if 'None' not in str(samplerType):
      modifDict = self._samplersDictionary[samplerType](**Kwargs)
      modified = parser.modifyOrAdd(modifDict)
    #newInputFiles = copy.deepcopy(currentInputFiles)
    #if type(Kwargs['prefix']) in [str,type("")]:#Specifing string type for python 2 and 3
    #  newInputFiles[index].setBase(Kwargs['prefix']+"~"+newInputFiles[index].getBase())
    #else:
    #  newInputFiles[index].setBase(str(Kwargs['prefix'][1][0])+'~'+newInputFiles[index].getBase())
    parser.printInput(currentInputFiles[index].getAbsFile(), modified)
    return currentInputFiles

  def monteCarloForRELAP7(self,**Kwargs):
    """
      This method is used to create a list of dictionaries that can be interpreted by the input Parser
      in order to change the input file based on the information present in the Kwargs dictionary.
      This is specific for Monte Carlo sampler
      @ In, **Kwargs, dict, kwared dictionary containing the values of the parameters to be changed
      @ Out, listDict, list, list of dictionaries used by the parser to change the input file
    """
    if 'prefix' in Kwargs:
      counter = Kwargs['prefix']
    else:
      raise IOError('a counter is needed for the Monte Carlo sampler for RELAP7')
    if 'initialSeed' in Kwargs:
      initSeed = Kwargs['initialSeed']
    else:
      initSeed = 1
    _,listDict = self.__genBasePointSampler(**Kwargs)
    #listDict = []
    modifDict = {}
    modifDict['name'] = ['Distributions', 'RNG_seed']
    RNGSeed = int(counter) + int(initSeed) - 1
    modifDict['RNG_seed'] = str(RNGSeed)
    listDict.append(modifDict)
    return listDict

  def dynamicEventTreeForRELAP7(self,**Kwargs):
    """
      This method is used to create a list of dictionaries that can be interpreted by the input Parser
      in order to change the input file based on the information present in the Kwargs dictionary.
      This is specific for DET sampler
      @ In, **Kwargs, dict, kwared dictionary containing the values of the parameters to be changed
      @ Out, listDict, list, list of dictionaries used by the parser to change the input file
    """
    listDict = []
    if 'hybridsamplerCoordinate' in Kwargs.keys():
      for preconditioner in Kwargs['hybridsamplerCoordinate']:
        preconditioner['executable'] = Kwargs['executable']
        if 'MonteCarlo' in preconditioner['SamplerType']:
          listDict = self.__genBasePointSampler(**preconditioner)[1]
          listDict.extend(self.monteCarloForRELAP7(**preconditioner))
        elif 'Grid' in preconditioner['SamplerType']:
          listDict.extend(self.gridForRELAP7(**preconditioner))
        elif 'Stratified' in preconditioner['SamplerType'] or 'Stratified' in preconditioner['SamplerType']:
          listDict.extend(self.latinHyperCubeForRELAP7(**preconditioner))
    # Check the initiator distributions and add the next threshold
    if 'initiatorDistribution' in Kwargs.keys():
      for i in range(len(Kwargs['initiatorDistribution'])):
        modifDict = {}
        varName = Kwargs['initiatorDistribution'][i]
        modifDict['name'] = ['Distributions', varName, 'ProbabilityThreshold']
        modifDict['ProbabilityThreshold'] = Kwargs['PbThreshold'][i]
        listDict.append(modifDict)
        del modifDict
    # add the initial time for this new branch calculation
    if 'startTime' in Kwargs.keys():
      if Kwargs['startTime'] != -sys.float_info.max:
        modifDict = {}
        startTime = Kwargs['startTime']
        modifDict['name'] = ['Executioner', 'start_time']
        modifDict['start_time'] = startTime
        listDict.append(modifDict)
        del modifDict
    # create the restart file name root from the parent branch calculation
    # in order to restart the calc from the last point in time
    if 'endTimeStep' in Kwargs.keys():
      #if Kwargs['endTimeStep'] != 0 or Kwargs['endTimeStep'] == 0:

      if Kwargs['startTime'] !=  -sys.float_info.max:
        modifDict = {}
        endTimeStepString = str(Kwargs['endTimeStep'])
        if(Kwargs['endTimeStep'] <= 9999):
          numZeros = 4 - len(endTimeStepString)
          for i in range(numZeros):
            endTimeStepString = "0" + endTimeStepString
        splitted = Kwargs['outfile'].split('~')
        output_parent = splitted[0] + '~'  + splitted[1]
        restartFileBase = os.path.join("..",utils.toString(Kwargs['RAVEN_parentID']),output_parent + "_cp",endTimeStepString)
        modifDict['name'] = ['Executioner']
        modifDict['restart_file_base'] = restartFileBase
        #print(' Restart file name base is "' + restart_file_base + '"')
        listDict.append(modifDict)
        del modifDict
    # max simulation time (if present)
    if 'endTime' in Kwargs.keys():
      modifDict = {}
      endTime = Kwargs['endTime']
      modifDict['name'] = ['Executioner']
      modifDict['end_time'] = endTime
      listDict.append(modifDict)
      del modifDict

    # in this way we erase the whole block in order to neglect eventual older info
    # remember this "command" must be added before giving the info for refilling the block
    modifDict = {}
    modifDict['name'] = ['RestartInitialize']
    modifDict['special'] = set(['erase_block'])
    listDict.append(modifDict)
    del modifDict
    # check and add the variables that have been changed by a distribution trigger
    # add them into the RestartInitialize block
    if 'branchChangedParam' in Kwargs.keys():
      if Kwargs['branchChangedParam'][0] not in ('None',b'None',None):
        for i in range(len(Kwargs['branchChangedParam'])):
          modifDict = {}
          modifDict['name'] = ['RestartInitialize',Kwargs['branchChangedParam'][i]]
          modifDict['value'] = Kwargs['branchChangedParamValue'][i]
          listDict.append(modifDict)
          del modifDict
    return listDict

  def __genBasePointSampler(self,**Kwargs):
    """
      Figure out which distributions need to be handled by
      the grid or Stratified samplers by modifying distributions in the .i file.
      Let the regular moose point sampler take care of the rest.
      Returns (distributions,listDict) where listDict is the
      start of the listDict that tells how to modify the input, and
      distributions is a dictionary with keys that are the 'variable name'
      and values of [computedValue,distribution name in .i file]
      Note that the key has "<distribution>" in front of the variable name.
      The actual variable can be gotten from the full key by:
      key[len('<distribution>'):]
      TODO This should check that the distributions in the .i file (if
      they exist) are consistent with the ones in the .xml file.
      TODO For variables, it should add them to the .csv file.
      @ In, **Kwargs, dict, kwared dictionary containing the values of the parameters to be changed
      @ Out, returnTuple, tuple, returnTuple[0] distributions dictionaries returnTuple[0] modified dictionary
    """
    distributionKeys = [key for key in Kwargs["SampledVars"] if key.startswith("<distribution>")]
    distributions = {}
    for key in distributionKeys:
      distributionName = Kwargs['distributionName'][key]
      distributionType = Kwargs['distributionType'][key]
      crowDistribution = json.loads(Kwargs['crowDist'][key])
      distributions[key] = [Kwargs["SampledVars"].pop(key),distributionName, distributionType,crowDistribution]
    from ..MooseBasedApp import MooseBasedAppInterface as mooseInterface
    # mooseInterface = utils.importFromPath(os.path.join(os.path.join(uppath(os.path.dirname(__file__),1),'MooseBasedApp'),'MooseBasedAppInterface.py'),False)
    mooseApp = mooseInterface.MooseBasedApp()
    returnTuple = distributions, mooseApp.pointSamplerForMooseBasedApp(**Kwargs)
    return returnTuple

  def gridForRELAP7(self,**Kwargs):
    """
      This method is used to create a list of dictionaries that can be interpreted by the input Parser
      in order to change the input file based on the information present in the Kwargs dictionary.
      This is specific for Grid sampler.
      Uses point sampler to generate variable points, and
      modifies distributions to be a zerowidth (constant) distribution
      at the grid point.
      @ In, **Kwargs, dict, kwared dictionary containing the values of the parameters to be changed
      @ Out, listDict, list, list of dictionaries used by the parser to change the input file
    """
    distributions,listDict = self.__genBasePointSampler(**Kwargs)
    for key in distributions.keys():
      distName, distType, crowDist = distributions[key][1:4]
      crowDist['name'] = ['Distributions',distName]
      #The following code would check more, but requires floating compare
      # that currently doesn't work properly
      #assertDict = crowDist.copy()
      #assertDict['special'] = set(['assert_match'])
      #listDict.append(assertDict)
      for crowDistKey in crowDist.keys():
        if crowDistKey not in ['type']:
          listDict.append({'name':['Distributions',distName], 'special':set(['assert_match']), crowDistKey:crowDist[crowDistKey]})

      listDict.append({'name':['Distributions',distName],
                       'special':set(['assert_match']),
                       'type':crowDist['type']})
      listDict.append({'name':['Distributions',distName],'special':set(['erase_block'])})
      listDict.append({'name':['Distributions',distName],'force_value':distributions[key][0]})
      listDict.append(crowDist)
    #print("listDict",listDict,"distributions",distributions,"Kwargs",Kwargs)
    return listDict

  def latinHyperCubeForRELAP7(self,**Kwargs):
    """
      This method is used to create a list of dictionaries that can be interpreted by the input Parser
      in order to change the input file based on the information present in the Kwargs dictionary.
      This is specific for Stratified sampler
      Uses point sampler to generate variable points, and truncates
      distribution to be inside of the latin hyper cube upper and lower
      bounds.
      @ In, **Kwargs, dict, kwared dictionary containing the values of the parameters to be changed
      @ Out, listDict, list, list of dictionaries used by the parser to change the input file
    """
    distributions,listDict = self.__genBasePointSampler(**Kwargs)
    for key in distributions.keys():
      distName, distType, crowDist = distributions[key][1:4]
      crowDist['name'] = ['Distributions',distName]
      #The following code would check more, but requires floating compare
      # that currently doesn't work properly
      #assertDict = crowDist.copy()
      #assertDict['special'] = set(['assert_match'])
      #listDict.append(assertDict)
      listDict.append({'name':['Distributions',distName],
                       'special':set(['assert_match']),
                       'type':crowDist['type']})
      listDict.append({'name':['Distributions',distName],
                       'special':set(['erase_block'])})
      listDict.append({'name':['Distributions',distName],
                       'V_window_Up':Kwargs['upper'][key]})
      listDict.append({'name':['Distributions',distName],
                       'V_window_Low':Kwargs['lower'][key]})
      listDict.append(crowDist)
    #print("listDict",listDict,"distributions",distributions)
    return listDict
