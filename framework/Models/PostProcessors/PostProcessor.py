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
Base class for PostProcessors
Created on March 3, 2021

@author: wangc
"""

#External Modules------------------------------------------------------------------------------------
import os
import abc
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from Models import Model
from Decorators.Parallelization import Parallel
#Internal Modules End--------------------------------------------------------------------------------

class PostProcessor(Model):
  """
    PostProcessor is an Action System. All the models here, take an input and perform an action
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(PostProcessor, cls).getInputSpecification()
    return inputSpecification

  @classmethod
  def generateValidateDict(cls):
    """
      This method generate a independent copy of validateDict for the calling class
      @ In, None
      @ Out, None
    """
    super(PostProcessor, cls).generateValidateDict()

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    cls.validateDict.pop('Sampler', None)
    cls.validateDict.pop('Optimizer', None)
    #the possible inputs
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][-1]['class'       ] = 'Databases'
    cls.validateDict['Input'  ][-1]['type'        ] = ['HDF5']
    cls.validateDict['Input'  ][-1]['required'    ] = False
    cls.validateDict['Input'  ][-1]['multiplicity'] = 'n'
    ## datasets
    dataObjects = cls.validateDict['Input'][0]
    dataObjects['type'].append('DataSet')
    # Cross validations will accept Model.ROM
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][-1]['class'       ] = 'Models'
    cls.validateDict['Input'  ][-1]['type'        ] = ['ROM']
    cls.validateDict['Input'  ][-1]['required'    ] = False
    cls.validateDict['Input'  ][-1]['multiplicity'] = 'n'
    #Some metrics can handle distributions
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][-1]['class'       ] = 'Distributions'
    cls.validateDict['Input'  ][-1]['type'        ] = ['']
    cls.validateDict['Input'  ][-1]['required'    ] = False
    cls.validateDict['Input'  ][-1]['multiplicity'] = 'n'
    #the possible outputs
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][-1]['class'       ] = 'Files'
    cls.validateDict['Output' ][-1]['type'        ] = ['']
    cls.validateDict['Output' ][-1]['required'    ] = False
    cls.validateDict['Output' ][-1]['multiplicity'] = 'n'
    # The possible functions
    cls.validateDict['Function'] = [cls.testDict.copy()]
    cls.validateDict['Function'  ][0]['class'       ] = 'Functions'
    cls.validateDict['Function'  ][0]['type'        ] = ['External','Internal']
    cls.validateDict['Function'  ][0]['required'    ] = False
    cls.validateDict['Function'  ][0]['multiplicity'] = 1

  def __init__(self, runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Model.__init__(self,runInfoDict)
    self.inputCheckInfo  = []     # List of tuple, i.e input objects info [('name','type')]
    self.action = None            # action
    self.workingDir = ''          # path for working directory
    self.printTag = 'POSTPROCESSOR MODEL'
    self.validDataType = ['PointSet','HistorySet'] # The list of accepted types of DataObject

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    Model._handleInput(self, paramInput)

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize the PostProcessor
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    self.inputs = inputs
    if 'stepName' in runInfo:
      self.workingDir = os.path.join(runInfo['WorkingDir'],runInfo['stepName']) #generate current working dir
    else:
      self.workingDir = runInfo['WorkingDir']
    self.inputCheckInfo = [(inp.name, inp.type) for inp in inputs]

  def createNewInput(self,myInput,samplerType,**kwargs):
    """
      This function will return a new input to be submitted to the postprocesor.
      (Not used but required by model base class)
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, myInput, list, the inputs (list) to start from to generate the new one
    """
    return myInput

  def inputToInternal(self, currentInput):
    """
      Method to convert an input object into the internal format that is
      understandable by the PostProcessor.
      @ In, currentInput, object, an object that needs to be converted
      @ Out, inputToInternal, list, list of current inputs
    """
    return [(copy.deepcopy(currentInput))]

  @abc.abstractmethod
  def run(self, input):
    """
      This method executes the postprocessor action.
      @ In,  input, object, object containing the data to process.
      Should avoid to use (inputToInternal output), and passing xarray directly/dataset
      Possible inputs include: dict, xarray.Dataset, pd.DataFrame
      @ Out, dict, xarray.Dataset, pd.DataFrame --> I think we can avoid collectoutput in the plugin pp
    """
    pass

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
    Input = self.createNewInput(myInput,samplerType, **kwargs)
    if Input is not None and len(Input) == 0:
      Input = None
    returnValue = (Input, self.run(Input))
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
    kwargs['forceThreads'] = True
    Model.submit(self,myInput, samplerType, jobHandler,**kwargs)

  def collectOutput(self,finishedjob,output,options=None):
    """
      Method that collects the outputs from the "run" method of the PostProcessor
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, optional, not used in PostProcessor.
        dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    if output.type not in self.validDataType:
      self.raiseAnError(IOError, 'Output type', str(output.type), 'is not allowed!')
    outputCheckInfo = (output.name, output.type)
    if outputCheckInfo in self.inputCheckInfo:
      self.raiseAnError(IOError, 'DataObject',output.name,'is used as both input and output of', \
              self.interface.printTag, 'This is not allowed! Please use different DataObjet as output')

    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError, "No available output to collect (run possibly not finished yet)")
    outputRealization = evaluation[1]

    if output.type in ['PointSet','HistorySet']:
      if self.outputDataset:
        self.raiseAnError(IOError, "DataSet output is required, but the provided type of DataObject is", output.type)
      self.raiseADebug('Dumping output in data object named ' + output.name)
      output.addRealization(outputRealization)
    elif output.type in ['DataSet']:
      self.raiseADebug('Dumping output in DataSet named ' + output.name)
      output.load(outputRealization,style='dataset')
