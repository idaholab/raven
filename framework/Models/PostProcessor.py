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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import Runners
from Models import Model
from Decorators.Parallelization import Parallel
from utils import utils, InputTypes
from .PostProcessors import factory as interfaceFactory
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
      @ Out, spec, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    spec = super().getInputSpecification()
    validTypes = list(interfaceFactory.knownTypes())
    typeEnum = InputTypes.makeEnumType('PostProcessor', 'PostProcessorType', validTypes)
    for name in validTypes:
      pp = interfaceFactory.returnClass(name)
      subSpec = pp.getInputSpecification()
      spec.mergeSub(subSpec)
    return spec

  @classmethod
  def generateValidateDict(cls):
    """
      This method generate a independent copy of validateDict for the calling class
      @ In, None
      @ Out, None
    """
    super().generateValidateDict()

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

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.inputCheckInfo  = []     # List of tuple, i.e input objects info [('name','type')]
    self.action = None            # action
    self.printTag = 'POSTPROCESSOR MODEL'
    self._pp = None

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    super()._readMoreXML(xmlNode)
    self._pp = interfaceFactory.returnInstance(self.subType)
    self._pp._readMoreXML(xmlNode)

  def whatDoINeed(self):
    """
      This method is used mainly by the Simulation class at the Step construction stage.
      It is used for inquiring the class, which is implementing the method, about the kind of objects the class needs to
      be initialize.
      @ In, None
      @ Out, needDict, dict, dictionary of objects needed (class:tuple(object type{if None,
        Simulation does not check the type}, object name))
    """
    needDict = super().whatDoINeed()
    needDictInterface = self._pp.whatDoINeed()
    needDict.update(needDictInterface)
    return needDict

  def generateAssembler(self, initDict):
    """
      This method is used mainly by the Simulation class at the Step construction stage.
      It is used for sending to the instanciated class, which is implementing the method,
      the objects that have been requested through "whatDoINeed" method
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):
        {specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    super().generateAssembler(initDict)
    self._pp.generateAssembler(initDict)

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize the PostProcessor
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    super().initialize(runInfo, inputs, initDict)
    self._pp.initialize(runInfo, inputs, initDict)
    self.inputCheckInfo = [(inp.name, inp.type) for inp in inputs]

  def createNewInput(self, myInput, samplerType, **kwargs):
    """
      This function will return a new input to be submitted to the postprocesor.
      (Not used but required by model base class)
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, passing through (consistent with base class but not used)
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, myInput, list, the inputs (list) to start from to generate the new one
    """
    return myInput

  @Parallel()
  def evaluateSample(self, myInput, samplerType, kwargs):
    """
        This will evaluate an individual sample on this model. Note, parameters
        are needed by createNewInput and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, passing through (consistent with base class but not used)
        @ In, kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
        @ Out, returnValue, tuple, This will hold two pieces of information,
          the first item will be the input data used to generate this sample,
          the second item will be the output of this model given the specified
          inputs
    """
    ppInput = self.createNewInput(myInput, samplerType, **kwargs)
    if ppInput is not None and len(ppInput) == 0:
      ppInput = None
    returnValue = (ppInput, self._pp.run(ppInput))
    return returnValue

  def submit(self,myInput,samplerType,jobHandler,**kwargs):
    """
        This will submit an individual sample to be evaluated by this model to a
        specified jobHandler. Note, some parameters are needed by createNewInput
        and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, passing through (consistent with base class but not used)
        @ In,  jobHandler, JobHandler instance, the global job handler instance
        @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
        @ Out, None
    """
    kwargs['forceThreads'] = True
    super().submit(myInput, samplerType, jobHandler,**kwargs)

  def collectOutput(self, finishedJob, output):
    """
      Method that collects the outputs from the "run" method of the PostProcessor
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ Out, None
    """
    outputCheckInfo = (output.name, output.type)
    if outputCheckInfo in self.inputCheckInfo:
      self.raiseAnError(IOError, 'DataObject',output.name,'is used as both input and output of', \
              self.interface.printTag, 'This is not allowed! Please use different DataObjet as output')

    self._pp.collectOutput(finishedJob, output)

  def provideExpectedMetaKeys(self):
     """
       Overrides the base class method to assure child postprocessor is also polled for its keys.
       @ In, None
       @ Out, meta, tuple, (set(str),dict), expected keys (empty if none) and the indexes related to expected keys
     """
     # get keys as per base class
     metaKeys,metaParams = super().provideExpectedMetaKeys()
     # add postprocessor keys
     keys, params = self._pp.provideExpectedMetaKeys()
     metaKeys = metaKeys.union(keys)
     metaParams.update(params)
     return metaKeys, metaParams
