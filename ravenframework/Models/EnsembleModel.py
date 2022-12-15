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
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules----------------------------------------------------------------------------------
import io
import sys
import copy
import numpy as np
import time
import itertools
from collections import OrderedDict
from ..Decorators.Parallelization import Parallel
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Dummy import Dummy
from ..utils import utils, InputData
from ..utils import graphStructure
from ..Runners import Error as rerror
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

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.modelsDictionary       = {}                    # dictionary of models that are going to be assembled
                                                        # {'modelName':{'Input':[in1,in2,..,inN],'Output':[out1,out2,..,outN],'Instance':Instance}}
    self.modelsInputDictionary  = {}                    # to allow reusability of ensemble modes (similar in construction to self.modelsDictionary)
    self.activatePicard         = False                 # is non-linear system being identified?
    self.localTargetEvaluations = {}                    # temporary storage of target evaluation data objects
    self.maxIterations          = 30                    # max number of iterations (in case of non-linear system activated)
    self.convergenceTol         = 1.e-3                 # tolerance of the iteration scheme (if activated) => L2 norm
    self.initialConditions      = {}                    # dictionary of initial conditions in case non-linear system is detected
    self.initialStartModels     = []                    # list of models that will execute first.
    self.ensembleModelGraph     = None                  # graph object (graphStructure.graphObject)
    self.printTag               = 'EnsembleModel MODEL' # print tag
    self.parallelStrategy = 1                           # parallel strategy [1=MPI like (internalParallel), 2=threads]
    self.runInfoDict = None                             # dictionary containing run info in case of parallelStrategy=2
    # assembler objects to be requested
    self.addAssemblerObject('Model', InputData.Quantity.one_to_infinity)
    self.addAssemblerObject('TargetEvaluation', InputData.Quantity.one_to_infinity)
    self.addAssemblerObject('Input', InputData.Quantity.one_to_infinity)
    self.addAssemblerObject('Output', InputData.Quantity.zero_to_infinity)

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
        # mirror the structure of medolsDictionary and modelsInputDictionary
        self.modelsInputDictionary[modelName] = {'TargetEvaluation':None,'Instance':None,'Input':[],'Output':[],'metadataToTransfer':[]}
        self.modelsDictionary[modelName] = {'TargetEvaluation':None,'Instance':None,'Input':[],'Output':[],'metadataToTransfer':[]}
        # number of allowed entries
        allowedEntriesLen = len(self.modelsInputDictionary[modelName].keys())
        for childChild in child:
          if childChild.tag.strip() == 'metadataToTransfer':
            # metadata that needs to be transfered from a source model into this model
            # list(metadataToTranfer, ModelSource,Alias (optional))
            if 'source' not in childChild.attrib.keys():
              self.raiseAnError(IOError, 'when metadataToTransfer XML block is defined, the "source" attribute must be inputted!')
            self.modelsInputDictionary[modelName][childChild.tag].append([childChild.text.strip(),childChild.attrib['source'],childChild.attrib.get("alias",None)])
          else:
            try:
              self.modelsInputDictionary[modelName][childChild.tag].append(childChild.text.strip())
            except AttributeError:
              self.modelsInputDictionary[modelName][childChild.tag] = childChild.text.strip()
            except KeyError:
              self.raiseAnError(IOError, 'The role '+str(childChild.tag) +" can not be used in the EnsebleModel. Check the manual for allowable nodes!")
        if list(self.modelsInputDictionary[modelName].values()).count(None) != 1:
          self.raiseAnError(IOError, "TargetEvaluation xml block needs to be inputted!")
        if len(self.modelsInputDictionary[modelName]['Input']) == 0:
          self.raiseAnError(IOError, "Input XML node for Model" + modelName +" has not been inputted!")
        if len(self.modelsInputDictionary[modelName].values()) > allowedEntriesLen:
          self.raiseAnError(IOError, "TargetEvaluation, Input and metadataToTransfer XML blocks are the only XML sub-blocks allowed!")
        if child.attrib['type'].strip() == "Code":
          self.createWorkingDir = True
      if child.tag == 'settings':
        self.__readSettings(child)
    if len(self.modelsInputDictionary.keys()) < 2:
      self.raiseAnError(IOError, "The EnsembleModel needs at least 2 models to be constructed!")
    for modelName in self.modelsInputDictionary.keys():
      if len(self.modelsInputDictionary[modelName]['Output']) == 0:
        self.modelsInputDictionary[modelName]['Output'] = None

  def __readSettings(self, xmlNode):
    """
      Method to read the ensemble model settings from XML input files
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'maxIterations':
        self.maxIterations  = int(child.text)
      elif child.tag == 'tolerance':
        self.convergenceTol = float(child.text)
      elif child.tag == 'initialStartModels':
        self.initialStartModels = list(inp.strip() for inp in child.text.strip().split(','))
      elif child.tag == 'initialConditions':
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
      Method to find the matching models with respect to some input/output. If not found, return None
      @ In, what, string, "Input" or "Output"
      @ In, subWhat, string, a keyword that needs to be contained in "what" for the matching model
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
      @ In, runInfo, dict, is the run info from the jobHandler
      @ In, inputs, list, is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
      @ Out, None
    """
    # store the job ids for jobs that we haven't collected optional output from
    # collect name of all the outputs in the Step
    outputsNames = []
    if initDict is not None:
      outputsNames = [output.name for output in initDict['Output']]

    # here we check if all the inputs inputted in the Step containing the EnsembleModel are actually used
    checkDictInputsUsage = dict((inp,False) for inp in inputs)
    # flag to check if a Code is in the Ensemble
    isThereACode = False
    # collect the models
    self.allOutputs = set()
    for modelClass, modelType, modelName, modelInstance in self.assemblerDict['Model']:
      if not isThereACode:
        isThereACode = modelType == 'Code'
      self.modelsDictionary[modelName]['Instance'] = modelInstance
      inputInstancesForModel = []
      for inputName in self.modelsInputDictionary[modelName]['Input']:
        inputInstancesForModel.append(self.retrieveObjectFromAssemblerDict('Input',inputName))
        checkDictInputsUsage[inputInstancesForModel[-1]] = True
      self.modelsDictionary[modelName]['InputObject'] = inputInstancesForModel

      # retrieve 'Output' objects, such as DataObjects, Databases to check if they are present in the Step
      if self.modelsInputDictionary[modelName]['Output'] is not None:
        outputNamesModel = []
        for output in self.modelsInputDictionary[modelName]['Output']:
          outputObject = self.retrieveObjectFromAssemblerDict('Output',output, True)
          if outputObject.name not in outputsNames:
            self.raiseAnError(IOError, "The optional Output "+outputObject.name+" listed for Model "+modelName+" is not present among the Step outputs!!!")
          outputNamesModel.append(outputObject.name)
        self.modelsDictionary[modelName]['OutputObject'] = outputNamesModel
      else:
        self.modelsDictionary[modelName]['OutputObject'] = []

      # initialize model
      self.modelsDictionary[modelName]['Instance'].initialize(runInfo,inputInstancesForModel,initDict)
      # retrieve 'TargetEvaluation' DataObjects
      targetEvaluation = self.retrieveObjectFromAssemblerDict('TargetEvaluation',self.modelsInputDictionary[modelName]['TargetEvaluation'], True)
      # assert acceptable TargetEvaluation types are used
      if targetEvaluation.type not in ['PointSet','HistorySet','DataSet']:
        self.raiseAnError(IOError, "Only DataObjects are allowed as TargetEvaluation object. Got "+ str(targetEvaluation.type)+"!")
      # localTargetEvaluations are for passing data and then resetting, not keeping data between samples
      self.localTargetEvaluations[modelName] = copy.deepcopy(targetEvaluation)
      # get input variables
      inps   = targetEvaluation.getVars('input')
      # get pivot parameters in input space if any and add it in the 'Input' list
      inDims = set([item for subList in targetEvaluation.getDimensions(var="input").values() for item in subList])
      # assemble the two lists
      self.modelsDictionary[modelName]['Input'] = inps + list(inDims - set(inps))
      # get output variables
      outs = targetEvaluation.getVars("output")
      # get pivot parameters in output space if any and add it in the 'Output' list
      outDims = set([item for subList in targetEvaluation.getDimensions(var="output").values() for item in subList])
      ## note, if a dimension is in both the input space AND output space, consider it an input
      outDims = outDims - inDims
      newOuts = outs + list(set(outDims) - set(outs))
      self.modelsDictionary[modelName]['Output'] = newOuts
      self.allOutputs = self.allOutputs.union(newOuts)
    # END loop to collect models
    self.allOutputs = list(self.allOutputs)
    if isThereACode:
      # FIXME: LEAVE IT HERE...WE NEED TO MODIFY HOW THE CODE GET RUN INFO...IT NEEDS TO BE ENCAPSULATED
      ## collect some run info
      ## self.runInfoDict = runInfo
      ## self.runInfoDict['numberNodes'] = runInfo.get('Nodes',[])
      ## check if MPI is activated
      ##if runInfo.get('NumMPI', 1) > 1:
      ##  self.parallelStrategy = 2 #  threads (OLD method)
      self.parallelStrategy = 2 #  threads (OLD method)

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
    allPath = self.ensembleModelGraph.findAllUniquePaths(self.initialStartModels)
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
      if len(self.initialStartModels) == 0:
        self.raiseAnError(IOError, "The 'initialStartModels' xml node is missing, this is required since the Picard's iteration is activated!")
      if len(self.initialConditions.keys()) == 0:
        self.raiseAnError(IOError,"Picard's iterations mode activated but no initial conditions provided!")
    else:
      if len(self.initialStartModels) !=0:
        self.raiseAnError(IOError, "The 'initialStartModels' xml node is not needed for non-Picard calculations, since the running sequence can be automatically determined by the code! Please delete this node to avoid a mistake.")
      self.raiseAMessage("EnsembleModel connections determined a linear system. Picard's iterations not activated!")

    for modelIn in self.modelsDictionary.keys():
      # in case there are metadataToTransfer, let's check if the source model is executed before the one that requests info
      if self.modelsInputDictionary[modelIn]['metadataToTransfer']:
        indexModelIn = self.orderList.index(modelIn)
        for metadataToGet, source, _ in self.modelsInputDictionary[modelIn]['metadataToTransfer']:
          if self.orderList.index(source) >= indexModelIn:
            self.raiseAnError(IOError, 'In model "'+modelIn+'" the "metadataToTransfer" named "'+metadataToGet+
                                       '" is linked to the source"'+source+'" that will be executed after this model.')
    self.needToCheckInputs = True
    # write debug statements
    self.raiseADebug("Specs of Graph Network represented by EnsembleModel:")
    self.raiseADebug("Graph Degree Sequence is    : "+str(self.ensembleModelGraph.degreeSequence()))
    self.raiseADebug("Graph Minimum/Maximum degree: "+str( (self.ensembleModelGraph.minDelta(), self.ensembleModelGraph.maxDelta())))
    self.raiseADebug("Graph density/diameter      : "+str( (self.ensembleModelGraph.density(),  self.ensembleModelGraph.diameter())))

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

  def __selectInputSubset(self,modelName, kwargs):
    """
      Method aimed to select the input subset for a certain model
      @ In, modelName, string, the model name
      @ In, kwargs , dict, the kwarded dictionary where the sampled vars are stored
      @ Out, selectedkwargs , dict, the subset of variables (in a swallow copy of the kwargs  dict)
    """
    selectedkwargs = copy.copy(kwargs)
    selectedkwargs['SampledVars'] = {}
    selectedkwargs['SampledVarsPb'] = {}
    for key in kwargs["SampledVars"].keys():
      if key in self.modelsDictionary[modelName]['Input']:
        selectedkwargs['SampledVars'][key]   = kwargs["SampledVars"][key]
        selectedkwargs['SampledVarsPb'][key] = kwargs["SampledVarsPb"][key] if 'SampledVarsPb' in kwargs.keys() and key in kwargs["SampledVarsPb"].keys() else 1.0
    return selectedkwargs

  def createNewInput(self,myInput,samplerType,**kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler or optimizer that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, newInputs, dict, dict that returns the new inputs for each sub-model
    """
    # check if all the inputs of the submodule are covered by the sampled vars and Outputs of the other sub-models
    if self.needToCheckInputs:
      allCoveredVariables = list(set(itertools.chain(self.allOutputs,kwargs['SampledVars'].keys())))

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
    outcomes, targetEvaluations, optionalOutputs = evaluation[1]
    joinedResponse = {}
    joinedGeneralMetadata = {}
    targetEvaluationNames = {}
    optionalOutputNames = {}
    joinedIndexMap = {} # collect all the index maps, then we can keep the ones we want?
    for modelIn in self.modelsDictionary.keys():
      targetEvaluationNames[self.modelsDictionary[modelIn]['TargetEvaluation']] = modelIn
      # collect data
      newIndexMap = outcomes[modelIn]['response'].get('_indexMap', None)
      if newIndexMap:
        joinedIndexMap.update(newIndexMap[0])
      joinedResponse.update(outcomes[modelIn]['response'])
      joinedGeneralMetadata.update(outcomes[modelIn]['general_metadata'])
      # collect the output of the STEP
      optionalOutputNames.update({outName : modelIn for outName in self.modelsDictionary[modelIn]['OutputObject']})
    # the prefix is re-set here
    joinedResponse['prefix'] = np.asarray([finishedJob.identifier])
    if joinedIndexMap:
      joinedResponse['_indexMap'] = np.atleast_1d(joinedIndexMap)

    if output.name not in optionalOutputNames:
      if output.name not in targetEvaluationNames.keys():
        # in the event a batch is run, the evaluations will be a dict as {'RAVEN_isBatch':True, 'realizations': [...]}
        if isinstance(evaluation,dict) and evaluation.get('RAVEN_isBatch',False):
          for rlz in evaluation['realizations']:
            output.addRealization(rlz)
        else:
          output.addRealization(joinedResponse)
      else:
        output.addRealization(outcomes[targetEvaluationNames[output.name]]['response'])
    else:
      # collect optional output if present and not already collected
      output.addRealization(optionalOutputs[optionalOutputNames[output.name]])

  def getAdditionalInputEdits(self,inputInfo):
    """
      Collects additional edits for the sampler to use when creating a new input. In this case, it calls all the getAdditionalInputEdits methods
      of the sub-models
      @ In, inputInfo, dict, dictionary in which to add edits
      @ Out, None.
    """
    for modelIn in self.modelsDictionary.keys():
      self.modelsDictionary[modelIn]['Instance'].getAdditionalInputEdits(inputInfo)

  @Parallel()
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
    kwargsToKeep = { keepKey: kwargs[keepKey] for keepKey in list(kwargs.keys())}
    jobHandler = kwargs['jobHandler'] if self.parallelStrategy == 2 else None
    Input = self.createNewInput(myInput[0], samplerType, **kwargsToKeep)

    ## Unpack the specifics for this class, namely just the jobHandler
    returnValue = (Input,self._externalRun(Input, jobHandler))
    return returnValue

  def submit(self,myInput,samplerType,jobHandler,**kwargs):
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
    prefix = kwargs['prefix']

    ## Ensemble models need access to the job handler, so let's stuff it in our
    ## catch all kwargs where evaluateSample can pick it up, not great, but
    ## will suffice until we can better redesign this whole process.
    kwargs['jobHandler'] = jobHandler if self.parallelStrategy == 2 else None
    ## This may look a little weird, but due to how the parallel python library
    ## works, we are unable to pass a member function as a job because the
    ## pp library loses track of what self is, so instead we call it from the
    ## class and pass self in as the first parameter

    nRuns = 1
    batchMode =  kwargs.get("batchMode", False)
    if batchMode:
      nRuns = kwargs["batchInfo"]['nRuns']

    for index in range(nRuns):
      if batchMode:
        kw =  kwargs['batchInfo']['batchRealizations'][index]
      else:
        kw = kwargs

      prefix = kw.get("prefix")
      uniqueHandler = kw.get("uniqueHandler",'any')
      forceThreads = kw.get("forceThreads",False)

      metadata = kw

      if self.parallelStrategy == 1:
        jobHandler.addJob((self, myInput, samplerType, kw), self.__class__.evaluateSample, prefix, metadata=metadata,
                  uniqueHandler=uniqueHandler, forceUseThreads=forceThreads,
                  groupInfo={'id': kwargs['batchInfo']['batchId'], 'size': nRuns} if batchMode else None)
      else:
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
      indexMap = previousOutputs.get('_indexMap', [{}])[0]
      if len(previousOutputs.values()) > 0:
        for inKey in self.modelsDictionary[modelIn]['Input']:
          if inKey in previousOutputs.keys():
            dependentOutputs[inKey] = previousOutputs[inKey] if len(previousOutputs[inKey]) > 1 else previousOutputs[inKey][0]
            indices = indexMap.get(inKey, None)
            if indices:
              if '_indexMap' not in dependentOutputs:
                dependentOutputs['_indexMap'] = {}
              dependentOutputs['_indexMap'][inKey] = indices
    return dependentOutputs

  def _externalRun(self,inRun, jobHandler = None):#, jobHandler):
    """
      Method that performs the actual run of the ensemble model (separated from run method for parallelization purposes)
      @ In, inRun, tuple, tuple of Inputs, e.g. inRun[0]: actual dictionary of input, inRun[1]: string,
        the type of Sampler or Optimizer, inRun[2], dict, contains the information from the Sampler
      @ In, jobHandler, object, optional, instance of jobHandler (available if parallelStrategy==2)
      @ Out, returnEvaluation, tuple, the results of the assembled model:
                               - returnEvaluation[0] dict of results from each sub-model,
                               - returnEvaluation[1] the dataObjects where the projection of each model is stored
                               - returnEvaluation[2] dict used to store the optional outputs
    """
    originalInput = inRun[0]
    samplerType = inRun[1]
    inputKwargs = inRun[2]
    identifier = inputKwargs.pop('prefix')
    tempOutputs = {}
    inRunTargetEvaluations = {}

    for modelIn in self.orderList:
      # reset the DataObject for the projection
      self.localTargetEvaluations[modelIn].reset()
      # deepcopy assures distinct copies
      inRunTargetEvaluations[modelIn] = copy.deepcopy(self.localTargetEvaluations[modelIn])
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
        # clear the model's Target Evaluation data object
        # in case there are metadataToTransfer, let's collect them from the source
        metadataToTransfer = None
        if self.modelsInputDictionary[modelIn]['metadataToTransfer']:
          metadataToTransfer = {}
        for metadataToGet, source, alias in self.modelsInputDictionary[modelIn]['metadataToTransfer']:
          if metadataToGet in returnDict[source]['general_metadata']:
            metaDataValue = returnDict[source]['general_metadata'][metadataToGet]
            metaDataValue = metaDataValue[0] if len(metaDataValue) == 1 else metaDataValue
            metadataToTransfer[metadataToGet if alias is None else alias] = metaDataValue
          elif metadataToGet in returnDict[source]['response']:
            metaDataValue = returnDict[source]['response'][metadataToGet]
            metaDataValue = metaDataValue[0] if len(metaDataValue) == 1 else metaDataValue
            metadataToTransfer[metadataToGet if alias is None else alias] = metaDataValue
          else:
            self.raiseAnError(RuntimeError,'metadata "'+metadataToGet+'" is not present among the ones available in source "'+source+'"!')
        # get dependent outputs
        dependentOutput = self.__retrieveDependentOutput(modelIn, gotOutputs, typeOutputs)
        # if nonlinear system, check for initial coditions
        if iterationCount == 1  and self.activatePicard:
          sampledVars = inputKwargs[modelIn]['SampledVars'].keys()
          conditionsToCheck = set(self.modelsDictionary[modelIn]['Input']) - set(itertools.chain(dependentOutput.keys(),sampledVars))
          for initialConditionToSet in conditionsToCheck:
            if initialConditionToSet in self.initialConditions.keys():
              dependentOutput[initialConditionToSet] = self.initialConditions[initialConditionToSet]
            else:
              self.raiseAnError(IOError,"No initial conditions provided for variable "+ initialConditionToSet)
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
        ## FIXME: this will come after we rework the "runInfo" collection in the code
        ## if run info is present, we need to pass to to kwargs
        ##if self.runInfoDict and 'Code' == self.modelsDictionary[modelIn]['Instance'].type:
        ##  inputKwargs[modelIn].update(self.runInfoDict)

        retDict, gotOuts, evaluation = self.__advanceModel(identifier, self.modelsDictionary[modelIn],
                                                        originalInput[modelIn], inputKwargs[modelIn],
                                                        inRunTargetEvaluations[modelIn], samplerType,
                                                        iterationCount, jobHandler)

        returnDict[modelIn] = retDict
        typeOutputs[modelCnt] = inRunTargetEvaluations[modelIn].type
        gotOutputs[modelCnt] =  gotOuts
        tempOutputs[modelIn] = evaluation

        # if nonlinear system, compute the residue
        ## it looks like this is handling _indexMap, but it's not clear since there's not a way to test it (yet).
        if self.activatePicard:
          residueContainer[modelIn]['iterValues'][1] = copy.copy(residueContainer[modelIn]['iterValues'][0])
          for out in  inRunTargetEvaluations[modelIn].getVars("output"):
            residueContainer[modelIn]['iterValues'][0][out] = copy.copy(gotOutputs[modelCnt][out])
            if iterationCount == 1:
              residueContainer[modelIn]['iterValues'][1][out] = np.zeros(len(residueContainer[modelIn]['iterValues'][0][out]))
          for out in gotOutputs[modelCnt].keys():
            residueContainer[modelIn]['residue'][out] = abs(np.asarray(residueContainer[modelIn]['iterValues'][0][out]) - np.asarray(residueContainer[modelIn]['iterValues'][1][out]))
          residueContainer[modelIn]['Norm'] =  np.linalg.norm(np.asarray(list(residueContainer[modelIn]['iterValues'][1].values()))-np.asarray(list(residueContainer[modelIn]['iterValues'][0].values())))

      # if nonlinear system, check the total residue and convergence
      if self.activatePicard:
        iterZero = []
        iterOne = []
        for modelIn in self.orderList:
          iterZero += residueContainer[modelIn]['iterValues'][0].values()
          iterOne  += residueContainer[modelIn]['iterValues'][1].values()
        residueContainer['TotalResidue'] = np.linalg.norm(np.asarray(iterOne)-np.asarray(iterZero))
        self.raiseAMessage("Picard's Iteration Norm: "+ str(residueContainer['TotalResidue']))
        residualPass = residueContainer['TotalResidue'] <= self.convergenceTol
        # sometimes there can be multiple residual values
        if hasattr(residualPass,'__len__'):
          residualPass = all(residualPass)
        if residualPass:
          self.raiseAMessage("Picard's Iteration converged. Norm: "+ str(residueContainer['TotalResidue']))
          break
    returnEvaluation = returnDict, inRunTargetEvaluations, tempOutputs
    return returnEvaluation

  def __advanceModel(self, identifier, modelToExecute, origInputList, inputKwargs, inRunTargetEvaluations, samplerType, iterationCount, jobHandler = None):
    """
      This method is aimed to advance the execution of a sub-model and to collect the data using
      the realization
      @ In, identifier, str, current job identifier
      @ In, modelToExecute, super(Model), Model instance than needs to be advanced
      @ In, origInputList, list, list of model input
      @ In, inputKwargs, dict, dictionary of kwargs for this model
      @ In, inRunTargetEvaluations, DataObject, target evaluation for the model to advance
      @ In, samplerType, str, sampler Type
      @ In, iterationCount, int, iteration counter (1 if not picard)
      @ In, jobHandler, jobHandler instance, optional, jobHandler instance (available only if parallelStrategy == 2)
      @ Out, returnDict, dict, dictionary containing the data extracted from the target evaluation
      @ Out, gotOutputs, dict, dictionary containing all the data coming out the model
      @ Out, evaluation, dict, the evaluation dictionary with the "unprojected" data
    """
    returnDict = {}

    self.raiseADebug('Submitting model',modelToExecute['Instance'].name)
    localIdentifier =  modelToExecute['Instance'].name+utils.returnIdSeparator()+identifier
    if self.parallelStrategy == 1:
      # we evaluate the model directly
      try:
        evaluation = modelToExecute['Instance'].evaluateSample.original_function(modelToExecute['Instance'], origInputList, samplerType, inputKwargs)
      except Exception as e:
        excType, excValue, excTrace = sys.exc_info()
        evaluation = None
    else:
      moveOn = False
      while not moveOn:
        # run the model
        inputKwargs.pop("jobHandler", None)
        modelToExecute['Instance'].submit(origInputList, samplerType, jobHandler, **inputKwargs)
        ## wait until the model finishes, in order to get ready to run the subsequential one
        while not jobHandler.isThisJobFinished(localIdentifier):
          time.sleep(1.e-3)
        moveOn = True
      # get job that just finished to gather the results
      finishedRun = jobHandler.getFinished(jobIdentifier = localIdentifier, uniqueHandler=self.name+identifier)
      evaluation = finishedRun[0].getEvaluation()
      if isinstance(evaluation, rerror):
        evaluation = None
        excType, excValue, excTrace = finishedRun[0].exceptionTrace
        e = rerror
        # the model failed
        for modelToRemove in list(set(self.orderList) - set([modelToExecute['Instance'].name])):
          jobHandler.getFinished(jobIdentifier = modelToRemove + utils.returnIdSeparator() + identifier, uniqueHandler = self.name + identifier)
      else:
        # collect the target evaluation
        modelToExecute['Instance'].collectOutput(finishedRun[0],inRunTargetEvaluations)

    if not evaluation:
      # the model failed
      import traceback
      msg = io.StringIO()
      traceback.print_exception(excType, excValue, excTrace, limit=10, file=msg)
      msg = msg.getvalue().replace('\n', '\n        ')
      self.raiseAnError(RuntimeError, f'The Model "{modelToExecute["Instance"].name}" id "{localIdentifier}" '+
                        f'failed! Trace:\n{"*"*72}\n{msg}\n{"*"*72}')
    else:
      if self.parallelStrategy == 1:
        inRunTargetEvaluations.addRealization(evaluation)
      else:
        modelToExecute['Instance'].collectOutput(finishedRun[0],inRunTargetEvaluations)

    ## FIXME: The call asDataset() is unuseful here. It must be done because otherwise the realization(...) method from collector
    ## does not return the indexes values (TO FIX)
    inRunTargetEvaluations.asDataset()
    # get realization
    dataSet = inRunTargetEvaluations.realization(index=iterationCount-1,unpackXArray=True)
    ##FIXME: the following dict construction is a temporary solution since the realization method returns scalars if we have a PointSet
    dataSet = {key:np.atleast_1d(dataSet[key]) for key in dataSet}
    responseSpace         = dataSet
    gotOutputs  = {key: dataSet[key] for key in inRunTargetEvaluations.getVars("output") + inRunTargetEvaluations.getVars("indexes")}
    if '_indexMap' in dataSet.keys():
      gotOutputs['_indexMap'] = dataSet['_indexMap']

    #store the results in return dictionary
    # store the metadata
    returnDict['response'        ] = copy.deepcopy(evaluation) #  this deepcopy must stay! alfoa
    # overwrite with target evaluation filtering
    returnDict['response'        ].update(responseSpace)
    returnDict['prefix'          ] = np.atleast_1d(identifier)
    returnDict['general_metadata'] = inRunTargetEvaluations.getMeta(general=True)

    return returnDict, gotOutputs, evaluation
