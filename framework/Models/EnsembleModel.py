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
Module where the base class and the specialization of different type of Model are
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import copy
import numpy as np
import time
from collections import OrderedDict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Dummy import Dummy
from utils import utils
from utils import graphStructure
import Runners
#Internal Modules End--------------------------------------------------------------------------------

class EnsembleModel(Dummy):
  """
    EnsembleModel class. This class is aimed to create a comunication 'pipe' among different models in terms of Input/Output relation
  """
  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      Being this class an essembler class, all the Inputs
      @ In, None
      @ Out, None
    """
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][1]['class'       ] = 'DataObjects'
    cls.validateDict['Output' ][1]['type'        ] = ['PointSet']
    cls.validateDict['Output' ][1]['required'    ] = False
    cls.validateDict['Output' ][1]['multiplicity'] = 'n'
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][2]['class'       ] = 'Databases'
    cls.validateDict['Output' ][2]['type'        ] = ['HDF5']
    cls.validateDict['Output' ][2]['required'    ] = False
    cls.validateDict['Output' ][2]['multiplicity'] = 'n'
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][3]['class'       ] = 'OutStreams'
    cls.validateDict['Output' ][3]['type'        ] = ['Plot','Print']
    cls.validateDict['Output' ][3]['required'    ] = False
    cls.validateDict['Output' ][3]['multiplicity'] = 'n'

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Dummy.__init__(self,runInfoDict)
    self.modelsDictionary      = {}       # dictionary of models that are going to be assembled
                                          # {'modelName':{'Input':[in1,in2,..,inN],'Output':[out1,out2,..,outN],'Instance':Instance}}
    self.activatePicard        = False    # is non-linear system beeing identified?
    self.tempTargetEvaluations = {}       # temporary storage of target evaluation data objects
    self.tempOutputs           = {}       # temporary storage of optional output data objects
    self.maxIterations         = 30       # max number of iterations (in case of non-linear system activated)
    self.convergenceTol        = 1.e-3    # tolerance of the iteration scheme (if activated) => L2 norm
    self.initialConditions     = {}       # dictionary of initial conditions in case non-linear system is detected
    self.ensembleModelGraph    = None     # graph object (graphStructure.graphObject)
    self.printTag = 'EnsembleModel MODEL' # print tag
    # assembler objects to be requested
    self.addAssemblerObject('Model','n',True)
    self.addAssemblerObject('TargetEvaluation','n')
    self.addAssemblerObject('Input','n')
    self.addAssemblerObject('Output','-n')

  def localInputAndChecks(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Dummy.localInputAndChecks(self, xmlNode)
    for child in xmlNode:
      if child.tag not in  ["Model","settings"]:
        self.raiseAnError(IOError, "Expected <Model> or <settings> tag. Got "+child.tag)
      if child.tag == 'Model':
        if 'type' not in child.attrib.keys() or 'class' not in child.attrib.keys():
          self.raiseAnError(IOError, 'Tag Model must have attributes "class" and "type"')
        # get model name
        modelName = child.text.strip()
        # create space of the allowed entries
        self.modelsDictionary[modelName] = {'TargetEvaluation':None,'Instance':None,'Input':[],'Output':[],'metadataToTransfer':[]}
        # number of allower entries
        allowedEntriesLen = len(self.modelsDictionary[modelName].keys())
        for childChild in child:
          if childChild.tag.strip() == 'metadataToTransfer':
            # metadata that needs to be transfered from a source model into this model
            # list(metadataToTranfer, ModelSource,Alias (optional))
            if 'source' not in childChild.attrib.keys():
              self.raiseAnError(IOError, 'when metadataToTransfer XML block is defined, the "source" attribute must be inputted!')
            self.modelsDictionary[modelName][childChild.tag].append([childChild.text.strip(),childChild.attrib['source'],childChild.attrib.get("alias",None)])
          else:
            try:
              self.modelsDictionary[modelName][childChild.tag].append(childChild.text.strip())
            except AttributeError:
              self.modelsDictionary[modelName][childChild.tag] = childChild.text.strip()
            except KeyError:
              self.raiseAnError(IOError, 'The role '+str(childChild.tag) +" can not be used in the EnsebleModel. Check the manual for allowable nodes!")
        if self.modelsDictionary[modelName].values().count(None) != 1:
          self.raiseAnError(IOError, "TargetEvaluation xml block needs to be inputted!")
        if len(self.modelsDictionary[modelName]['Input']) == 0:
          self.raiseAnError(IOError, "Input XML node for Model" + modelName +" has not been inputted!")
        if len(self.modelsDictionary[modelName].values()) > allowedEntriesLen:
          self.raiseAnError(IOError, "TargetEvaluation, Input and metadataToTransfer XML blocks are the only XML sub-blocks allowed!")
        if child.attrib['type'].strip() == "Code":
          self.createWorkingDir = True
      if child.tag == 'settings':
        self.__readSettings(child)
    if len(self.modelsDictionary.keys()) < 2:
      self.raiseAnError(IOError, "The EnsembleModel needs at least 2 models to be constructed!")
    for modelName in self.modelsDictionary.keys():
      if len(self.modelsDictionary[modelName]['Output']) == 0:
        self.modelsDictionary[modelName]['Output'] = None

  def __readSettings(self, xmlNode):
    """
      Method to read the ensemble model settings from XML input files
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'maxIterations':
        self.maxIterations  = int(child.text)
      if child.tag == 'tolerance':
        self.convergenceTol = float(child.text)
      if child.tag == 'initialConditions':
        for var in child:
          if "repeat" in var.attrib.keys():
            self.initialConditions[var.tag] = np.repeat([float(var.text.split()[0])], int(var.attrib['repeat'])) #np.array([float(var.text.split()[0]) for _ in range(int(var.attrib['repeat']))])
          else:
            try:
              values = var.text.split()
              self.initialConditions[var.tag] = float(values[0]) if len(values) == 1 else np.asarray([float(varValue) for varValue in values])
            except:
              self.raiseAnError(IOError,"unable to read text from XML node "+var.tag)

  def __findMatchingModel(self,what,subWhat):
    """
      Method to find the matching models with respect a some input/output. If not found, return None
      @ In, what, string, "Input" or "Output"
      @ In, subWhat, string, a keyword that needs to be contained in "what" for the mathching model
      @ Out, models, list, list of model names that match the key subWhat
    """
    models = []
    for key, value in self.modelsDictionary.items():
      if subWhat in value[what]:
        models.append(key)
    if len(models) == 0:
      models = None
    return models

  ##############################################################################
  # #To be uncommented when the execution list can be handled
  # def __getExecutionList(self, orderedNodes, allPath):
  #   """
  #    Method to get the execution list
  #    @ In, orderedNodes, list, list of models ordered based
  #                     on the input/output relationships
  #    @ In, allPath, list, list of lists containing all the
  #                     path from orderedNodes[0] to orderedNodes[-1]
  #    @ Out, executionList, list, list of lists with the execution
  #                     order ([[model1],[model2.1,model2.2],[model3], etc.]
  #   """
  #   numberPath    = len(allPath)
  #   maxComponents = max([len(path) for path in allPath])

  #   executionList = [ [] for _ in range(maxComponents)]
  #   executionCounter = -1
  #   for node in orderedNodes:
  #     nodeCtn = 0
  #   for path in allPath:
  #     if node in path:
  #       nodeCtn +=1
  #   if nodeCtn == numberPath:
  #     executionCounter+=1
  #     executionList[executionCounter] = [node]
  #   else:
  #     previousNodesInPath = []
  #     for path in allPath:
  #       if path.count(node) > 0:
  #         previousNodesInPath.append(path[path.index(node)-1])
  #     for previousNode in previousNodesInPath:
  #       if previousNode in executionList[executionCounter]:
  #         executionCounter+=1
  #         break
  #     executionList[executionCounter].append(node)
  #   return executionList
  ##############################################################################

  def initialize(self,runInfo,inputs,initDict=None):
    """
      Method to initialize the EnsembleModel
      @ In, runInfo is the run info from the jobHandler
      @ In, inputs is a list containing whatever is passed with an input role in the step
      @ In, initDict, optional, dictionary of all objects available in the step is using this model
      @ Out, None
    """
    # in here we store the job ids for which we did not collected the optional output yet
    self.tempOutputs['uncollectedJobIds'] = []
    self.tempOutputs['forHold'] = {}
    # collect name of all the outputs in the Step
    outputsNames = []
    if initDict is not None:
      outputsNames = [output.name for output in initDict['Output']]

    # here we check if all the inputs inputted in the Step containing the EnsembleModel are acttualy used
    checkDictInputsUsage = {}
    for input in inputs:
      checkDictInputsUsage[input] = False

    for modelIn in self.assemblerDict['Model']:
      self.modelsDictionary[modelIn[2]]['Instance'] = modelIn[3]
      inputInstancesForModel = []
      for input in self.modelsDictionary[modelIn[2]]['Input']:
        inputInstancesForModel.append( self.retrieveObjectFromAssemblerDict('Input',input))
        checkDictInputsUsage[inputInstancesForModel[-1]] = True
      self.modelsDictionary[modelIn[2]]['InputObject'] = inputInstancesForModel

      if self.modelsDictionary[modelIn[2]]['Output'] is not None:
        outputInstancesForModel = []
        for output in self.modelsDictionary[modelIn[2]]['Output']:
          outputObject = self.retrieveObjectFromAssemblerDict('Output',output)
          if outputObject.name not in outputsNames:
            self.raiseAnError(IOError, "The optional Output "+outputObject.name+" listed for Model "+modelIn[2]+" is not present among the Step outputs!!!")
          outputInstancesForModel.append( self.retrieveObjectFromAssemblerDict('Output',output))
        self.modelsDictionary[modelIn[2]]['OutputObject'] = outputInstancesForModel
      else:
        self.modelsDictionary[modelIn[2]]['OutputObject'] = []
      self.modelsDictionary[modelIn[2]]['Instance'].initialize(runInfo,inputInstancesForModel,initDict)
      for mm in self.modelsDictionary[modelIn[2]]['Instance'].mods:
        if mm not in self.mods:
          self.mods.append(mm)
      self.modelsDictionary[modelIn[2]]['TargetEvaluation'] = self.retrieveObjectFromAssemblerDict('TargetEvaluation',self.modelsDictionary[modelIn[2]]['TargetEvaluation'])
      if self.modelsDictionary[modelIn[2]]['TargetEvaluation'].type not in ['PointSet','HistorySet']:
        self.raiseAnError(IOError, "Only DataObjects are allowed as TargetEvaluation object. Got "+ str(self.modelsDictionary[modelIn[2]]['TargetEvaluation'].type)+"!")
      self.tempTargetEvaluations[modelIn[2]]                = copy.deepcopy(self.modelsDictionary[modelIn[2]]['TargetEvaluation'])
      self.modelsDictionary[modelIn[2]]['Input' ]           = self.modelsDictionary[modelIn[2]]['TargetEvaluation'].getParaKeys("inputs")
      self.modelsDictionary[modelIn[2]]['Output']           = self.modelsDictionary[modelIn[2]]['TargetEvaluation'].getParaKeys("outputs")
    # check if all the inputs passed in the step are linked with at least a model
    if not all(checkDictInputsUsage.values()):
      unusedFiles = ""
      for inFile, used in checkDictInputsUsage.items():
        if not used:
          unusedFiles+= " "+inFile.name
      self.raiseAnError(IOError, "The following inputs specified in the Step are not used in the EnsembleModel: "+unusedFiles)
    # construct chain connections
    modelsToOutputModels  = dict.fromkeys(self.modelsDictionary.keys(),None)
    # find matching models
    for modelIn in self.modelsDictionary.keys():
      outputMatch = []
      for i in range(len(self.modelsDictionary[modelIn]['Output'])):
        match = self.__findMatchingModel('Input',self.modelsDictionary[modelIn]['Output'][i])
        outputMatch.extend(match if match is not None else [])
      outputMatch = list(set(outputMatch))
      modelsToOutputModels[modelIn] = outputMatch
    # construct the ensemble model directed graph
    self.ensembleModelGraph = graphStructure.graphObject(modelsToOutputModels)
    # make some checks
    if not self.ensembleModelGraph.isConnectedNet():
      isolatedModels = self.ensembleModelGraph.findIsolatedVertices()
      self.raiseAnError(IOError, "Some models are not connected. Possible candidates are: "+' '.join(isolatedModels))
    # get all paths
    allPath = self.ensembleModelGraph.findAllUniquePaths()
    ###################################################
    # to be removed once executionList can be handled #
    self.orderList = self.ensembleModelGraph.createSingleListOfVertices(allPath)
    self.raiseAMessage("Model Execution list: "+' -> '.join(self.orderList))
    ###################################################
    ###########################################################################################
    # To be uncommented when the execution list can be handled                                #
    # if len(allPath) > 1: self.executionList = self.__getExecutionList(self.orderList,allPath) #
    # else               : self.executionList = allPath[-1]                                     #
    ###########################################################################################
    # check if Picard needs to be activated
    self.activatePicard = self.ensembleModelGraph.isALoop()
    if self.activatePicard:
      self.raiseAMessage("EnsembleModel connections determined a non-linear system. Picard's iterations activated!")
      if len(self.initialConditions.keys()) == 0:
        self.raiseAnError(IOError,"Picard's iterations mode activated but no intial conditions provided!")
    else:
      self.raiseAMessage("EnsembleModel connections determined a linear system. Picard's iterations not activated!")

    self.allOutputs = []
    for modelIn in self.modelsDictionary.keys():
      for modelInOut in self.modelsDictionary[modelIn]['Output']:
        if modelInOut not in self.allOutputs:
          self.allOutputs.append(modelInOut)
      # in case there are metadataToTransfer, let's check if the source model is executed before the one that requests info
      if self.modelsDictionary[modelIn]['metadataToTransfer']:
        indexModelIn = self.orderList.index(modelIn)
        for metadataToGet, source, _ in self.modelsDictionary[modelIn]['metadataToTransfer']:
          if self.orderList.index(source) >= indexModelIn:
            self.raiseAnError(IOError, 'In model "'+modelIn+'" the "metadataToTransfer" named "'+metadataToGet+
                                       '" is linked to the source"'+source+'" that will be executed after this model.')
    self.needToCheckInputs = True
    # write debug statements
    self.raiseAMessage("Specs of Graph Network represented by EnsembleModel:")
    self.raiseAMessage("Graph Degree Sequence is    : "+str(self.ensembleModelGraph.degreeSequence()))
    self.raiseAMessage("Graph Minimum/Maximum degree: "+str( (self.ensembleModelGraph.minDelta(), self.ensembleModelGraph.maxDelta())))
    self.raiseAMessage("Graph density/diameter      : "+str( (self.ensembleModelGraph.density(),  self.ensembleModelGraph.diameter())))

  def getInitParams(self):
    """
      Method used to export to the printer in the base class the additional PERMANENT your local class have
      @ In, None
      @ Out, tempDict, dict, dictionary to be updated. {'attribute name':value}
    """
    tempDict = OrderedDict()
    tempDict['Models contained in EnsembleModel are '] = self.modelsDictionary.keys()
    for modelIn in self.modelsDictionary.keys():
      tempDict['Model '+modelIn+' TargetEvaluation is '] = self.modelsDictionary[modelIn]['TargetEvaluation']
      tempDict['Model '+modelIn+' Inputs are '] = self.modelsDictionary[modelIn]['Input']
    return tempDict

  def getCurrentSetting(self):
    """
      Function to inject the name and values of the parameters that might change during the simulation
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys and each parameter's initial value as the dictionary values
    """
    paramDict = self.getInitParams()
    return paramDict

  def __selectInputSubset(self,modelName, kwargs ):
    """
      Method aimed to select the input subset for a certain model
      @ In, modelName, string, the model name
      @ In, kwargs , dict, the kwarded dictionary where the sampled vars are stored
      @ Out, selectedkwargs , dict, the subset of variables (in a swallow copy of the kwargs  dict)
    """
    selectedkwargs = copy.copy(kwargs)
    selectedkwargs['SampledVars'], selectedkwargs['SampledVarsPb'] = {}, {}
    for key in kwargs["SampledVars"].keys():
      if key in self.modelsDictionary[modelName]['Input']:
        selectedkwargs['SampledVars'][key], selectedkwargs['SampledVarsPb'][key] =  kwargs["SampledVars"][key],  kwargs["SampledVarsPb"][key] if 'SampledVarsPb' in kwargs.keys() else 1.0
    return selectedkwargs

  def createNewInput(self,myInput,samplerType,**kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, newInputs, dict, dict that returns the new inputs for each sub-model
    """
    # check if all the inputs of the submodule are covered by the sampled vars and Outputs of the other sub-models
    if self.needToCheckInputs:
      allCoveredVariables = list(set(self.allOutputs + kwargs['SampledVars'].keys()))

    identifier = kwargs['prefix']
    # global prefix
    newKwargs = {'prefix':identifier}

    newInputs = {}

    ## First check the inputs if they need to be checked
    if self.needToCheckInputs:
      for modelIn, specs in self.modelsDictionary.items():
        for inp in specs['Input']:
          if inp not in allCoveredVariables:
            self.raiseAnError(RuntimeError,"for sub-model "+ modelIn + " the input "+inp+" has not been found among other models' outputs and sampled variables!")

    ## Now prepare the new inputs for each model
    for modelIn, specs in self.modelsDictionary.items():
      newKwargs[modelIn] = self.__selectInputSubset(modelIn,kwargs)

      # if specs['Instance'].type != 'Code':
      #   inputDict = [self._inputToInternal(self.modelsDictionary[modelIn]['InputObject'][0],newKwargs['SampledVars'].keys())]
      # else:
      #   inputDict = self.modelsDictionary[modelIn]['InputObject']

      # local prefix
      newKwargs[modelIn]['prefix'] = modelIn+utils.returnIdSeparator()+identifier
      newInputs[modelIn]  = self.modelsDictionary[modelIn]['InputObject']

      # if specs['Instance'].type == 'Code':
      #   newInputs[modelIn][1]['originalInput'] = inputDict

    self.needToCheckInputs = False
    return (newInputs, samplerType, newKwargs)

  def collectOutput(self,finishedJob,output):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, ClientRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError,"Job " + finishedJob.identifier +" failed!")
    out = evaluation[1]
    exportDict = {'inputSpaceParams':{},'outputSpaceParams':{},'metadata':{}}
    exportDictTargetEvaluation = {}
    outcomes, targetEvaluations, optionalOutputs = out
    try:
      jobIndex = self.tempOutputs['uncollectedJobIds'].index(finishedJob.identifier)
      self.tempOutputs['uncollectedJobIds'].pop(jobIndex)
    except ValueError:
      jobIndex = None
    for modelIn in self.modelsDictionary.keys():
      # collect data
      inputsValues               = targetEvaluations[modelIn].getParametersValues('inputs', nodeId = 'RecontructEnding')
      unstructuredInputsValues   = targetEvaluations[modelIn].getParametersValues('unstructuredInputs', nodeId = 'RecontructEnding')
      outputsValues              = targetEvaluations[modelIn].getParametersValues('outputs', nodeId = 'RecontructEnding')
      metadataValues             = targetEvaluations[modelIn].getAllMetadata(nodeId = 'RecontructEnding')
      inputsValues  = inputsValues if targetEvaluations[modelIn].type != 'HistorySet' else inputsValues.values()[-1]
      if len(unstructuredInputsValues.keys()) > 0:
        if targetEvaluations[modelIn].type != 'HistorySet':
          castedUnstructuredInputsValues = {}
          for key in unstructuredInputsValues.keys():
            castedUnstructuredInputsValues[key] = unstructuredInputsValues[key][-1]
        else:
          castedUnstructuredInputsValues  =  unstructuredInputsValues.values()[-1]
        inputsValues.update(castedUnstructuredInputsValues)
      outputsValues  = outputsValues if targetEvaluations[modelIn].type != 'HistorySet' else outputsValues.values()[-1]
      exportDictTargetEvaluation[self.modelsDictionary[modelIn]['TargetEvaluation'].name] = {'inputSpaceParams':inputsValues,'outputSpaceParams':outputsValues,'metadata':metadataValues}
      for typeInfo,values in outcomes[modelIn].items():
        for key in values.keys():
          exportDict[typeInfo][key] = np.asarray(values[key])
      # collect optional output if present and not already collected
      if jobIndex is not None:
        for optionalModelOutput in self.modelsDictionary[modelIn]['OutputObject']:
          self.modelsDictionary[modelIn]['Instance'].collectOutput(finishedJob,optionalModelOutput,options={'exportDict':copy.copy(optionalOutputs[modelIn])})
    # collect the output of the STEP
    optionalOutputNames = []
    for modelIn in self.modelsDictionary.keys():
      for optionalOutput in self.modelsDictionary[modelIn]['OutputObject']:
        optionalOutputNames.append(optionalOutput.name)
    if output.type == 'HDF5':
      if output.name not in optionalOutputNames:
        output.addGroupDataObjects({'group':self.name+str(finishedJob.identifier)},exportDict,False)
    else:
      if output.name not in optionalOutputNames:
        if output.name in exportDictTargetEvaluation.keys():
          exportDict = exportDictTargetEvaluation[output.name]
        for key in exportDict['inputSpaceParams' ] :
          if key in output.getParaKeys('inputs'):
            output.updateInputValue (key,exportDict['inputSpaceParams' ][key])
        for key in exportDict['outputSpaceParams'] :
          if key in output.getParaKeys('outputs'):
            output.updateOutputValue(key,exportDict['outputSpaceParams'][key])
        for key in exportDict['metadata']:
          output.updateMetadata(key,exportDict['metadata'][key][-1])
    # collect outputs for "holding"
    # first clear old outputs
    # TODO FIXME this is a flawed implementation, since it requires that the "holdOutputErase" is of
    # a very specific form that works with the the current SPSA optimizer.  Since we have no other optimizer right now,
    # the problem is only extensibility, not the actual implementation.
    if exportDict['metadata'].get('holdOutputErase',None) is not None:
      keys = self.tempOutputs['forHold'].keys()
      toErase = exportDict['metadata']['holdOutputErase'][0].split('_')[:2]
      for key in keys:
        traj,itr,pert = key.split('_')
        if traj == toErase[0] and itr <= toErase[1]:
          del self.tempOutputs['forHold'][key]
    #then hold on to the current output
    #TODO we shouldn't be doing this unless the user asked us to hold outputs!  FIXME
    self.tempOutputs['forHold'][finishedJob.identifier] = {'outs':optionalOutputs,'targetEvaluations':targetEvaluations}

  def getAdditionalInputEdits(self,inputInfo):
    """
      Collects additional edits for the sampler to use when creating a new input. In this case, it calls all the getAdditionalInputEdits methods
      of the sub-models
      @ In, inputInfo, dict, dictionary in which to add edits
      @ Out, None.
    """
    for modelIn in self.modelsDictionary.keys():
      self.modelsDictionary[modelIn]['Instance'].getAdditionalInputEdits(inputInfo)

  def evaluateSample(self, myInput, samplerType, kwargs):
    """
        This will evaluate an individual sample on this model. Note, parameters
        are needed by createNewInput and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, is the type of sampler that is calling to generate a new input
        @ In, kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
        @ Out, returnValue, dict, This holds the output information of the evaluated sample.
    """
    jobHandler = kwargs.pop('jobHandler')
    Input = self.createNewInput(myInput[0], samplerType, **kwargs)

    ## Unpack the specifics for this class, namely just the jobHandler
    returnValue = (Input,self._externalRun(Input,jobHandler))
    return returnValue

  def submit(self,myInput,samplerType,jobHandler,**kwargs):
    """
        This will submit an individual sample to be evaluated by this model to a
        specified jobHandler. Note, some parameters are needed by createNewInput
        and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, is the type of sampler that is calling to generate a new input
        @ In,  jobHandler, JobHandler instance, the global job handler instance
        @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
        @ Out, None
    """
    for mm in utils.returnImportModuleString(jobHandler):
      if mm not in self.mods:
        self.mods.append(mm)

    prefix = kwargs['prefix']
    self.tempOutputs['uncollectedJobIds'].append(prefix)

    ## Ensemble models need access to the job handler, so let's stuff it in our
    ## catch all kwargs where evaluateSample can pick it up, not great, but
    ## will suffice until we can better redesign this whole process.
    kwargs['jobHandler'] = jobHandler

    ## This may look a little weird, but due to how the parallel python library
    ## works, we are unable to pass a member function as a job because the
    ## pp library loses track of what self is, so instead we call it from the
    ## class and pass self in as the first parameter
    jobHandler.addJob((self, myInput, samplerType, kwargs), self.__class__.evaluateSample, prefix, kwargs)


  def submitAsClient(self,myInput,samplerType,jobHandler,**kwargs):
    """
        This will submit an individual sample to be evaluated by this model to a
        specified jobHandler as a client job. Note, some parameters are needed
        by createNewInput and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new
          one
        @ In, samplerType, string, is the type of sampler that is calling to
          generate a new input
        @ In,  jobHandler, JobHandler instance, the global job handler instance
        @ In, **kwargs, dict,  is a dictionary that contains the information
          coming from the sampler, a mandatory key is the sampledVars' that
          contains a dictionary {'name variable':value}
        @ Out, None
    """
    for mm in utils.returnImportModuleString(jobHandler):
      if mm not in self.mods:
        self.mods.append(mm)

    prefix = kwargs['prefix']
    self.tempOutputs['uncollectedJobIds'].append(prefix)

    ## Ensemble models need access to the job handler, so let's stuff it in our
    ## catch all kwargs where evaluateSample can pick it up, not great, but
    ## will suffice until we can better redesign this whole process.
    kwargs['jobHandler'] = jobHandler

    ## This may look a little weird, but due to how the parallel python library
    ## works, we are unable to pass a member function as a job because the
    ## pp library loses track of what self is, so instead we call it from the
    ## class and pass self in as the first parameter
    jobHandler.addClientJob((self, myInput, samplerType, kwargs), self.__class__.evaluateSample, prefix, kwargs)

  def __retrieveDependentOutput(self,modelIn,listOfOutputs, typeOutputs):
    """
      This method is aimed to retrieve the values of the output of the models on which the modelIn depends on
      @ In, modelIn, string, name of the model for which the dependent outputs need to be
      @ In, listOfOutputs, list, list of dictionary outputs ({modelName:dictOfOutputs})
      @ Out, dependentOutputs, dict, the dictionary of outputs the modelIn needs
    """
    dependentOutputs = {}
    for previousOutputs, outputType in zip(listOfOutputs,typeOutputs):
      if len(previousOutputs.values()) > 0:
        for input in self.modelsDictionary[modelIn]['Input']:
          if input in previousOutputs.keys():
            dependentOutputs[input] =  previousOutputs[input][-1] if outputType != 'HistorySet' else np.asarray(previousOutputs[input])
          #if input in previousOutputs.keys(): dependentOutputs[input] =  previousOutputs[input] if outputType != 'HistorySet' else np.asarray(previousOutputs[input])
    return dependentOutputs

  def _identifyModelsOnHold(self,holdOutputSpace):
    """
      This method is aimed to identify the models that belong to a requested "on hold" outputspace
      @ In, holdOutputSpace, list, list of output variables whose models should be kept on hold
      @ Out, modelsOnHold, list, list of Models on hold
    """
    modelsOnHold = []
    modelsOutputBool = {}

    for modelIn in self.modelsDictionary:
      modelsOutputBool[modelIn] = {key:False for key in self.modelsDictionary[modelIn]['Output']}
      for outputVar in holdOutputSpace:
        if outputVar in self.modelsDictionary[modelIn]['Output']:
          modelsOnHold.append(modelIn)
          modelsOutputBool[modelIn][outputVar] = True
    modelsOnHold = list(set(modelsOnHold))
    for holdModel in modelsOnHold:
      if len(set(modelsOutputBool[holdModel].values())) > 1:
        self.raiseAnError(RuntimeError,"In order to keep on hold a model, all the outputs generated by that model must be kept on hold!"+"Model: "+ holdModel)
    return modelsOnHold

  def _externalRun(self,inRun, jobHandler):
    """
      Method that performs the actual run of the essembled model (separated from run method for parallelization purposes)
      @ In, inRun, tuple, tuple of Inputs (inRun[0] actual input, inRun[1] jobHandler instance )
      @ Out, returnEvaluation, tuple, the results of the essembled model:
                               - returnEvaluation[0] dict of results from each sub-model,
                               - returnEvaluation[1] the dataObjects where the projection of each model is stored
    """
    originalInput = inRun[0]
    samplerType = inRun[1]
    inputKwargs = inRun[2]
    identifier = inputKwargs.pop('prefix')
    tempOutputs = {}
    tempTargetEvaluations = {}
    holdOutputSpace = inputKwargs.values()[-1]['holdOutputSpace'] if 'holdOutputSpace' in inputKwargs.values()[-1] else None
    # the sampler or optimizer wants to hold the result of
    modelsOnHold    = []
    holdCollector   = {}
    if holdOutputSpace is not None:
      modelsOnHold = self._identifyModelsOnHold(holdOutputSpace[0])
      for modelOnHold in modelsOnHold:
        holdCollector[modelOnHold] = {'exportDict':self.tempOutputs['forHold'][holdOutputSpace[1]]['outs'][modelOnHold],'targetEvaluations':self.tempOutputs['forHold'][holdOutputSpace[1]]['targetEvaluations'][modelOnHold]} #         self.tempOutputs['forHold'][holdOutputSpace[1]][modelOnHold]
    #    holdCollector[modelOnHold] = self.modelsDictionary[modelOnHold]['TargetEvaluation'].getRealizationGivenEvaluationID(holdOutputSpace[1])

    for modelIn in self.orderList:
      self.tempTargetEvaluations[modelIn].resetData()
      tempTargetEvaluations[modelIn] = copy.copy(self.tempTargetEvaluations[modelIn])
    residueContainer = dict.fromkeys(self.modelsDictionary.keys())
    gotOutputs       = [{}]*len(self.orderList)
    typeOutputs      = ['']*len(self.orderList)

    # if nonlinear system, initialize residue container
    if self.activatePicard:
      for modelIn in self.orderList:
        residueContainer[modelIn] = {'residue':{},'iterValues':[{}]*2}
        for out in self.modelsDictionary[modelIn]['Output']:
          residueContainer[modelIn]['residue'][out] = np.zeros(1)
          residueContainer[modelIn]['iterValues'][0][out] = np.zeros(1)
          residueContainer[modelIn]['iterValues'][1][out] = np.zeros(1)

    maxIterations = self.maxIterations if self.activatePicard else 1
    iterationCount = 0
    while iterationCount < maxIterations:
      returnDict     = {}
      iterationCount += 1

      if self.activatePicard:
        self.raiseAMessage("Picard's Iteration "+ str(iterationCount))

      for modelCnt, modelIn in enumerate(self.orderList):
        tempTargetEvaluations[modelIn].resetData()
        # in case there are metadataToTransfer, let's collect them from the source
        metadataToTransfer = None
        if self.modelsDictionary[modelIn]['metadataToTransfer']:
          metadataToTransfer = {}

        for metadataToGet, source, alias in self.modelsDictionary[modelIn]['metadataToTransfer']:
          if metadataToGet not in returnDict[source]['metadata'].keys():
            self.raiseAnError(RuntimeError,'metadata "'+metadataToGet+'" is not present among the ones available in source "'+source+'"!')
          metadataToTransfer[metadataToGet if alias is None else alias] = returnDict[source]['metadata'][metadataToGet][-1]

        # get dependent outputs
        dependentOutput = self.__retrieveDependentOutput(modelIn, gotOutputs, typeOutputs)
        # if nonlinear system, check for initial coditions
        if iterationCount == 1  and self.activatePicard:
          sampledVars = inputKwargs[modelIn]['SampledVars'].keys()
          conditionsToCheck = set(self.modelsDictionary[modelIn]['Input']) - set(dependentOutput.keys()+sampledVars)
          for initialConditionToSet in conditionsToCheck:
            if initialConditionToSet in self.initialConditions.keys():
              dependentOutput[initialConditionToSet] = self.initialConditions[initialConditionToSet]
            else:
              self.raiseAnError(IOError,"No initial conditions provided for variable "+ initialConditionToSet)

          ## Does the same as above, probably should see if either of these is faster,
          ## otherwise I would recommend the block above for its clarity.
          # for initCondToSet in [x for x in self.modelsDictionary[modelIn]['Input'] if x not in set(dependentOutput.keys()+sampledVars)]:
          #   if initCondToSet in self.initialConditions.keys():
          #     dependentOutput[initCondToSet] = self.initialConditions[initCondToSet]
          #   else:
          #     self.raiseAnError(IOError,"No initial conditions provided for variable "+ initCondToSet)

        # set new identifiers
        inputKwargs[modelIn]['prefix']        = modelIn+utils.returnIdSeparator()+identifier
        inputKwargs[modelIn]['uniqueHandler'] = self.name+identifier
        if metadataToTransfer is not None:
          inputKwargs[modelIn]['metadataToTransfer'] = metadataToTransfer

        for key, value in dependentOutput.items():
          inputKwargs[modelIn]["SampledVars"  ][key] =  dependentOutput[key]
          ## FIXME it is a mistake (Andrea). The SampledVarsPb for this variable should be transferred from outside
          ## Who has this information? -- DPM 4/11/17
          inputKwargs[modelIn]["SampledVarsPb"][key] =  1.0
        self._replaceVariablesNamesWithAliasSystem(inputKwargs[modelIn]["SampledVars"  ],'input',False)
        self._replaceVariablesNamesWithAliasSystem(inputKwargs[modelIn]["SampledVarsPb"],'input',False)

        nextModel = False
        while not nextModel:
          moveOn = False
          while not moveOn:
            if jobHandler.availability() > 0:
              # run the model
              if modelIn not in modelsOnHold:
                self.modelsDictionary[modelIn]['Instance'].submit(originalInput[modelIn], samplerType, jobHandler, **inputKwargs[modelIn])
                # wait until the model finishes, in order to get ready to run the subsequential one
                while not jobHandler.isThisJobFinished(modelIn+utils.returnIdSeparator()+identifier):
                  time.sleep(1.e-3)
              nextModel = moveOn = True
            else:
              time.sleep(1.e-3)
          # store the results in the working dictionaries
            returnDict[modelIn]   = {}
          if modelIn not in modelsOnHold:
            # get job that just finished to gather the results
            finishedRun = jobHandler.getFinished(jobIdentifier = modelIn+utils.returnIdSeparator()+identifier, uniqueHandler=self.name+identifier)
            evaluation = finishedRun[0].getEvaluation()
            if isinstance(evaluation, Runners.Error):
              # the model failed
              for modelToRemove in self.orderList:
                if modelToRemove != modelIn:
                  jobHandler.getFinished(jobIdentifier = modelToRemove + utils.returnIdSeparator() + identifier, uniqueHandler = self.name + identifier)
              self.raiseAnError(RuntimeError,"The Model  " + modelIn + " identified by " + finishedRun[0].identifier +" failed!")

            # collect output in the temporary data object
            exportDict = self.modelsDictionary[modelIn]['Instance'].createExportDictionaryFromFinishedJob(finishedRun[0], True)
          else:
            exportDict = holdCollector[modelIn]['exportDict']
          # store the output dictionary
          tempOutputs[modelIn] = copy.deepcopy(exportDict)

          # collect the target evaluation
          if modelIn not in modelsOnHold:
            self.modelsDictionary[modelIn]['Instance'].collectOutput(finishedRun[0],tempTargetEvaluations[modelIn],options={'exportDict':exportDict})
          else:
            tempTargetEvaluations[modelIn] = holdCollector[modelIn]['targetEvaluations']

          responseSpace         = tempTargetEvaluations[modelIn].getParametersValues('outputs', nodeId = 'RecontructEnding')
          inputSpace            = tempTargetEvaluations[modelIn].getParametersValues('inputs', nodeId = 'RecontructEnding')
          typeOutputs[modelCnt] = tempTargetEvaluations[modelIn].type
          gotOutputs[modelCnt]  = responseSpace if typeOutputs[modelCnt] != 'HistorySet' else responseSpace.values()[-1]

          #store the results in return dictionary
          returnDict[modelIn]['outputSpaceParams'] = gotOutputs[modelCnt]
          returnDict[modelIn]['inputSpaceParams' ] = inputSpace if typeOutputs[modelCnt] != 'HistorySet' else inputSpace.values()[-1]
          returnDict[modelIn]['metadata'         ] = tempTargetEvaluations[modelIn].getAllMetadata()

          # if nonlinear system, compute the residue
          if self.activatePicard:
            residueContainer[modelIn]['iterValues'][1] = copy.copy(residueContainer[modelIn]['iterValues'][0])
            for out in gotOutputs[modelCnt].keys():
              residueContainer[modelIn]['iterValues'][0][out] = copy.copy(gotOutputs[modelCnt][out])
              if iterationCount == 1:
                residueContainer[modelIn]['iterValues'][1][out] = np.zeros(len(residueContainer[modelIn]['iterValues'][0][out]))
            for out in gotOutputs[modelCnt].keys():
              residueContainer[modelIn]['residue'][out] = abs(np.asarray(residueContainer[modelIn]['iterValues'][0][out]) - np.asarray(residueContainer[modelIn]['iterValues'][1][out]))
            residueContainer[modelIn]['Norm'] =  np.linalg.norm(np.asarray(residueContainer[modelIn]['iterValues'][1].values())-np.asarray(residueContainer[modelIn]['iterValues'][0].values()))

      # if nonlinear system, check the total residue and convergence
      if self.activatePicard:
        iterZero = []
        iterOne = []
        for modelIn in self.orderList:
          iterZero += residueContainer[modelIn]['iterValues'][0].values()
          iterOne  += residueContainer[modelIn]['iterValues'][1].values()
        residueContainer['TotalResidue'] = np.linalg.norm(np.asarray(iterOne)-np.asarray(iterZero))
        self.raiseAMessage("Picard's Iteration Norm: "+ str(residueContainer['TotalResidue']))
        if residueContainer['TotalResidue'] <= self.convergenceTol:
          self.raiseAMessage("Picard's Iteration converged. Norm: "+ str(residueContainer['TotalResidue']))
          break
    returnEvaluation = returnDict, tempTargetEvaluations, tempOutputs
    return returnEvaluation

  def acceptHoldOutputSpace(self):
    """
      This method returns True if a certain output space can be kept on hold (so far, just the EnsembelModel can do that)
      @ In, None
      @ Out, acceptHoldOutputSpace, bool, True if a certain output space can be kept on hold
    """
    return True
