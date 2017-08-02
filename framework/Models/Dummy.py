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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Model import Model
from utils import utils
from utils.cached_ndarray import c1darray
import Runners
#Internal Modules End--------------------------------------------------------------------------------

class Dummy(Model):
  """
    This is a dummy model that just return the effect of the sampler. The values reported as input in the output
    are the output of the sampler and the output is the counter of the performed sampling
  """
  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Model.__init__(self,runInfoDict)
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
    cls.validateDict['Input' ][0]['type'        ] = ['PointSet']
    cls.validateDict['Input' ][0]['required'    ] = True
    cls.validateDict['Input' ][0]['multiplicity'] = 1
    cls.validateDict['Output'][0]['type'        ] = ['PointSet']

  def _manipulateInput(self,dataIn):
    """
      Method that is aimed to manipulate the input in order to return a common input understandable by this class
      @ In, dataIn, object, the object that needs to be manipulated
      @ Out, inRun, dict, the manipulated input
    """
    if len(dataIn)>1:
      self.raiseAnError(IOError,'Only one input is accepted by the model type '+self.type+' with name '+self.name)
    if type(dataIn[0])!=tuple:
      inRun = self._inputToInternal(dataIn[0]) #this might happen when a single run is used and the input it does not come from self.createNewInput
    else:
      inRun = dataIn[0][0]
    return inRun

  def _inputToInternal(self,dataIN,full=False):
    """
      Transform it in the internal format the provided input. dataIN could be either a dictionary (then nothing to do) or one of the admitted data
      @ In, dataIn, object, the object that needs to be manipulated
      @ In, full, bool, optional, does the full input needs to be retrieved or just the last element?
      @ Out, localInput, dict, the manipulated input
    """
    #self.raiseADebug('wondering if a dictionary compatibility should be kept','FIXME')
    if  type(dataIN).__name__ !='dict':
      if dataIN.type not in self.admittedData:
        self.raiseAnError(IOError,self,'type "'+dataIN.type+'" is not compatible with the model "' + self.type + '" named "' + self.name+'"!')
    if type(dataIN)!=dict:
      localInput = dict.fromkeys(dataIN.getParaKeys('inputs' )+dataIN.getParaKeys('outputs' ),None)
      if not dataIN.isItEmpty():
        if dataIN.type == 'PointSet':
          for entries in dataIN.getParaKeys('inputs' ):
            localInput[entries] = copy.copy(np.array(dataIN.getParam('input' ,entries))[0 if full else -1:])
          for entries in dataIN.getParaKeys('outputs'):
            localInput[entries] = copy.copy(np.array(dataIN.getParam('output',entries))[0 if full else -1:])
        else:
          if full:
            for hist in range(len(dataIN)):
              realization = dataIN.getRealization(hist)
              for entries in dataIN.getParaKeys('inputs' ):
                if localInput[entries] is None:
                  localInput[entries] = c1darray(shape=(1,))
                localInput[entries].append(realization['inputs'][entries])
              for entries in dataIN.getParaKeys('outputs' ):
                if localInput[entries] is None:
                  localInput[entries] = []
                localInput[entries].append(realization['outputs'][entries])
          else:
            realization = dataIn.getRealization(len(dataIn)-1)
            for entries in dataIN.getParaKeys('inputs' ):
              localInput[entries] = [realization['inputs'][entries]]
            for entries in dataIN.getParaKeys('outputs' ):
              localInput[entries] = [realization['outputs'][entries]]

      #Now if an OutputPlaceHolder is used it is removed, this happens when the input data is not representing is internally manufactured
      if 'OutputPlaceHolder' in dataIN.getParaKeys('outputs'):
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
    if len(myInput)>1:
      self.raiseAnError(IOError,'Only one input is accepted by the model type '+self.type+' with name'+self.name)

    inputDict   = self._inputToInternal(myInput[0])
    self._replaceVariablesNamesWithAliasSystem(inputDict,'input',False)

    if 'SampledVars' in kwargs.keys():
      sampledVars = self._replaceVariablesNamesWithAliasSystem(kwargs['SampledVars'],'input',False)
      for key in kwargs['SampledVars'].keys():
        inputDict[key] = np.atleast_1d(kwargs['SampledVars'][key])

    for val in inputDict.values():
      if val is None:
        self.raiseAnError(IOError,'While preparing the input for the model '+self.type+' with name '+self.name+' found a None input variable '+ str(inputDict.items()))
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
    returnValue = (inRun,{'OutputPlaceHolder':np.atleast_1d(np.float(Input[1]['prefix']))})
    return returnValue

  def collectOutput(self,finishedJob,output,options=None):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    # create the export dictionary
    if options is not None and 'exportDict' in options:
      exportDict = options['exportDict']
    else:
      exportDict = self.createExportDictionaryFromFinishedJob(finishedJob)
    self.addOutputFromExportDictionary(exportDict, output, options, finishedJob.identifier)

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
    #check for name usage, depends on where it comes from
    if 'inputSpaceParams' in exportDict.keys():
      inKey = 'inputSpaceParams'
      outKey = 'outputSpaceParams'
    else:
      inKey = 'inputs'
      outKey = 'outputs'
    if not set(output.getParaKeys('inputs') + output.getParaKeys('outputs')).issubset(set(list(exportDict[inKey].keys()) + list(exportDict[outKey].keys()))):
      missingParameters = set(output.getParaKeys('inputs') + output.getParaKeys('outputs')) - set(list(exportDict[inKey].keys()) + list(exportDict[outKey].keys()))
      self.raiseAnError(RuntimeError,"the model "+ self.name+" does not generate all the outputs requested in output object "+ output.name +". Missing parameters are: " + ','.join(list(missingParameters)) +".")

    for key in output.getParaKeys('inputs'):
      if key in exportDict[inKey ]:
        output.updateInputValue(key,exportDict[inKey ][key],options)
      else:
        self.raiseAnError(Exception, "the input parameter "+key+" requested in the DataObject "+output.name+
                                  " has not been found among the Model input paramters ("+",".join(exportDict[inKey ].keys())+"). Check your input!")
    for key in output.getParaKeys('outputs'):
      if key in exportDict[outKey]:
        output.updateOutputValue(key,exportDict[outKey][key],options)
      else:
        self.raiseAnError(Exception, "the output parameter "+key+" requested in the DataObject "+output.name+
                                  " has not been found among the Model output paramters ("+",".join(exportDict[outKey].keys())+"). Check your input!")
    for key in exportDict['metadata']:
      output.updateMetadata(key,exportDict['metadata'][key])
