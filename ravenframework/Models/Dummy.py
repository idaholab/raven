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
#External Modules------------------------------------------------------------------------------------
import copy
import itertools
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Model import Model
from ..utils import utils
from ..utils.cached_ndarray import c1darray
from ..Decorators.Parallelization import Parallel
#Internal Modules End--------------------------------------------------------------------------------

class Dummy(Model):
  """
    This is a dummy model that just return the effect of the sampler. The values reported as input in the output
    are the output of the sampler and the output is the counter of the performed sampling
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.admittedData = self.__class__.validateDict['Input' ][0]['type'] #the list of admitted data is saved also here for run time checks
    #the following variable are reset at each call of the initialize method
    self.printTag = 'DUMMY MODEL'

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    cls.validateDict['Input' ]                    = [cls.validateDict['Input' ][0]]
    cls.validateDict['Input' ][0]['type'        ] = ['PointSet','DataSet']
    cls.validateDict['Input' ][0]['required'    ] = True
    cls.validateDict['Input' ][0]['multiplicity'] = 1
    cls.validateDict['Output'][0]['type'        ] = ['PointSet','DataSet']

  def _copyModel(self, obj):
    """
      Set this instance to be a copy of the provided object.
      This is used to replace placeholder models with serialized objects
      during deserialization in IOStep.
      @ In, obj, instance, the instance of the object to copy from
      @ Out, None
    """
    if obj.type != self.type:
      self.raiseAnError(IOError,'Only objects of the same type can be copied! {} != {} !'.format(obj.type, self.type))
    try:
      #If __getstate__ and __setstate__ available, use them
      self.__setstate__(obj.__getstate__())
    except AttributeError:
      #Otherwise just use object's dictionary
      self.__dict__.update(obj.__dict__)

  def _manipulateInput(self,dataIn):
    """
      Method that is aimed to manipulate the input in order to return a common input understandable by this class
      @ In, dataIn, object, the object that needs to be manipulated
      @ Out, inRun, dict, the manipulated input
    """
    if type(dataIn[0])!=tuple:
      inRun = self._inputToInternal(dataIn[0]) #this might happen when a single run is used and the input it does not come from self.createNewInput
    else:
      inRun = dataIn[0][0]
    return inRun

  def _inputToInternal(self,dataIN):
    """
      Transform it in the internal format the provided input. dataIN could be either a dictionary (then nothing to do) or one of the admitted data
      @ In, dataIn, object, the object that needs to be manipulated
      @ Out, localInput, dict, the manipulated input
    """
    # FIXME wondering if a dictionary compatibility should be kept - Dan M.
    if not isinstance(dataIN, dict):
      if dataIN.type not in self.admittedData:
        self.raiseAnError(IOError,self,'type "'+dataIN.type+'" is not compatible with the model "' + self.type + '" named "' + self.name+'"!')
    if not isinstance(dataIN, dict):
      localInput = dict.fromkeys(dataIN.getVars('input')+dataIN.getVars('output')+dataIN.indexes,None)
      if not len(dataIN) == 0:
        dataSet = dataIN.asDataset()
        if dataIN.type == 'PointSet':
          for entries in dataIN.getVars('input')+dataIN.getVars('output'):
            localInput[entries] = copy.copy(dataSet[entries].values)
        elif dataIN.type == 'HistorySet':
          # Andrea Explanation
          # This part of the code had to be speeded up
          # For large dataset ~1000 variables, 50 histories this was taking almost ~1 hr
          # I kept here "tolist" to keep compatibility but this
          # is 100000 faster for large datasets
          nsamples = len(dataSet.coords[dataIN.sampleTag])
          for index in dataIN.indexes:
            localInput[index] =  list(np.repeat(np.atleast_2d(dataSet[index].values), nsamples, axis=0))
          sizeIndex = len(localInput[index][-1])
          for o in dataIN.getVars('output'):
            localInput[o] =  list(dataSet[o].values)
          for entries in dataIN.getVars('input'):
            localInput[entries] =  list(np.repeat(np.atleast_2d(dataSet[entries].values).T, sizeIndex, axis=1))
        elif dataIN.type == 'DataSet':
          for rlz in range(len(dataIN)):
            for index in dataIN.indexes:
              if localInput[index] is None:
                localInput[index] = []
                selDict = {dataIN.sampleTag: rlz}
              localInput[index].append(dataSet.isel(**selDict)[index].values)
            for entry in dataIN.getVars('input') + dataIN.getVars('output'):
              if localInput[entry] is None:
                localInput[entry] = []
              value = dataSet.isel({dataIN.sampleTag: rlz})[entry].values
              localInput[entry].append(value)
      #Now if an OutputPlaceHolder is used it is removed, this happens when the input data is not representing is internally manufactured
      if 'OutputPlaceHolder' in dataIN.getVars('output'):
        localInput.pop('OutputPlaceHolder') # this remove the counter from the inputs to be placed among the outputs
    else:
      localInput = dataIN #here we do not make a copy since we assume that the dictionary is for just for the model usage and any changes are not impacting outside
    return localInput

  def createNewInput(self,myInput,samplerType,**kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      here only a PointSet is accepted a local copy of the values is performed.
      For a PointSet, only the last set of entries are copied
      The copied values are returned as a dictionary back
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, ([(inputDict)],copy.deepcopy(kwargs)), tuple, return the new input in a tuple form
    """
    inputDict   = self._inputToInternal(myInput[0])
    self._replaceVariablesNamesWithAliasSystem(inputDict,'input',False)

    if 'SampledVars' in kwargs.keys():
      sampledVars = self._replaceVariablesNamesWithAliasSystem(kwargs['SampledVars'],'input',False)
      for key in kwargs['SampledVars'].keys():
        inputDict[key] = np.atleast_1d(kwargs['SampledVars'][key])

    missing = list(var for var,val in inputDict.items() if val is None)
    if len(missing) != 0:
      self.raiseAnError(IOError,'Input values for variables {} not found while preparing the input for model "{}"!'.format(missing,self.name))
    #the inputs/outputs should not be store locally since they might be used as a part of a list of input for the parallel runs
    #same reason why it should not be used the value of the counter inside the class but the one returned from outside as a part of the input

    ## SampledVars should almost always be in the kwargs, but in the off chance
    ## it is not, we want to continue as normal. Rather than use an if, we do
    ## it this way, since the kwargs can have an arbitrary size of keys in it.
    try:
      if len(self.alias['input'].keys()) != 0:
        kwargs['SampledVars'] = sampledVars
    except KeyError:
      pass
    return [(inputDict)],copy.deepcopy(kwargs)

  @Parallel()
  def evaluateSample(self, myInput, samplerType, kwargs):
    """
        This will evaluate an individual sample on this model. Note, parameters
        are needed by createNewInput and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, is the type of sampler that is calling to generate a new input
        @ In, kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
        @ Out, returnValue, tuple, This will hold two pieces of information,
          the first item will be the input data used to generate this sample,
          the second item will be the output of this model given the specified
          inputs
    """
    Input = self.createNewInput(myInput, samplerType, **kwargs)
    inRun = self._manipulateInput(Input[0])
    # alias system
    self._replaceVariablesNamesWithAliasSystem(inRun,'input',True)
    self._replaceVariablesNamesWithAliasSystem(kwargs['SampledVars'],'input',True)
    # build realization using input space from inRun and metadata from kwargs
    rlz = dict((var,np.atleast_1d(inRun[var] if var in kwargs['SampledVars'] else kwargs[var])) for var in set(itertools.chain(kwargs.keys(),inRun.keys())))
    # add dummy output space
    rlz['OutputPlaceHolder'] = np.atleast_1d(float(Input[1]['prefix']))
    return rlz

  def collectOutput(self,finishedJob,output,options=None):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    # TODO START can be abstracted to base class
    # TODO apparently sometimes "options" can include 'exportDict'; what do we do for this?
    # TODO consistency with old HDF5; fix this when HDF5 api is in place
    # TODO expensive deepcopy prevents modification when sent to multiple outputs
    result = finishedJob.getEvaluation()
    # alias system
    self._replaceVariablesNamesWithAliasSystem(result,'output',True)
    output.addRealization(result)
    # END can be abstracted to base class

  def collectOutputFromDict(self,exportDict,output,options=None):
    """
      Collect results from a dictionary
      @ In, exportDict, dict, contains 'inputSpaceParams','outputSpaceParams','metadata'
      @ In, output, DataObject, to whom we write the data
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    #prefix is not generally useful for dummy-related models, so we remove it but store it
    if 'prefix' in exportDict.keys():
      prefix = exportDict.pop('prefix')
    if 'inputSpaceParams' in exportDict.keys():
      inKey = 'inputSpaceParams'
      outKey = 'outputSpaceParams'
    else:
      inKey = 'inputs'
      outKey = 'outputs'

    rlz = {}
    rlz.update(exportDict[inKey])
    rlz.update(exportDict[outKey])
    rlz.update(exportDict['metadata'])
    for k,v in rlz.items():
      rlz[k] = np.atleast_1d(v)
    output.addRealization(rlz)
    return
