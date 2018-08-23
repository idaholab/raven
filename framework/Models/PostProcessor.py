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
import os
import inspect
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Model import Model
from utils import utils
import PostProcessors
#Internal Modules End--------------------------------------------------------------------------------

class PostProcessor(Model):
  """
    PostProcessor is an Action System. All the models here, take an input and perform an action
  """
  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    cls.validateDict['Input']                    = [cls.validateDict['Input' ][0]]
    cls.validateDict['Input'][0]['required'    ] = False
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][1]['class'       ] = 'Databases'
    cls.validateDict['Input'  ][1]['type'        ] = ['HDF5']
    cls.validateDict['Input'  ][1]['required'    ] = False
    cls.validateDict['Input'  ][1]['multiplicity'] = 'n'
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][2]['class'       ] = 'DataObjects'
    cls.validateDict['Input'  ][2]['type'        ] = ['PointSet','HistorySet']
    cls.validateDict['Input'  ][2]['required'    ] = False
    cls.validateDict['Input'  ][2]['multiplicity'] = 'n'
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][3]['class'       ] = 'Files'
    # FIXME there's lots of types that Files can be, so until XSD replaces this, commenting this out
    #cls.validateDict['Input'  ][3]['type'        ] = ['']
    cls.validateDict['Input'  ][3]['required'    ] = False
    cls.validateDict['Input'  ][3]['multiplicity'] = 'n'
    # Cross validations will accept Model.ROM
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][4]['class'       ] = 'Models'
    cls.validateDict['Input'  ][4]['type'        ] = ['ROM']
    cls.validateDict['Input'  ][4]['required'    ] = False
    cls.validateDict['Input'  ][4]['multiplicity'] = 'n'
    #Some metrics can handle distributions
    cls.validateDict['Input'].append(cls.testDict.copy())
    cls.validateDict['Input'  ][5]['class'       ] = 'Distributions'
    cls.validateDict['Input'  ][5]['type'        ] = ['']
    cls.validateDict['Input'  ][5]['required'    ] = False
    cls.validateDict['Input'  ][5]['multiplicity'] = 'n'
    cls.validateDict['Output'].append(cls.testDict.copy())
    cls.validateDict['Output' ][0]['class'       ] = 'Files'
    cls.validateDict['Output' ][0]['type'        ] = ['']
    cls.validateDict['Output' ][0]['required'    ] = False
    cls.validateDict['Output' ][0]['multiplicity'] = 'n'
    cls.validateDict['Output' ][1]['class'       ] = 'DataObjects'
    cls.validateDict['Output' ][1]['type'        ] = ['PointSet','HistorySet','DataSet']
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
    cls.validateDict['Function'] = [cls.testDict.copy()]
    cls.validateDict['Function'  ][0]['class'       ] = 'Functions'
    cls.validateDict['Function'  ][0]['type'        ] = ['External','Internal']
    cls.validateDict['Function'  ][0]['required'    ] = False
    cls.validateDict['Function'  ][0]['multiplicity'] = 1
    cls.validateDict['ROM'] = [cls.testDict.copy()]
    cls.validateDict['ROM'       ][0]['class'       ] = 'Models'
    cls.validateDict['ROM'       ][0]['type'        ] = ['ROM']
    cls.validateDict['ROM'       ][0]['required'    ] = False
    cls.validateDict['ROM'       ][0]['multiplicity'] = 1
    cls.validateDict['KDD'] = [cls.testDict.copy()]
    cls.validateDict['KDD'       ][0]['class'       ] = 'Models'
    cls.validateDict['KDD'       ][0]['type'        ] = ['KDD']
    cls.validateDict['KDD'       ][0]['required'    ] = False
    cls.validateDict['KDD'       ][0]['multiplicity'] = 'n'

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Model.__init__(self,runInfoDict)
    self.input  = {}     # input source
    self.action = None   # action
    self.workingDir = ''
    self.printTag = 'POSTPROCESSOR MODEL'

  def provideExpectedMetaKeys(self):
    """
      Overrides the base class method to assure child postprocessor is also polled for its keys.
      @ In, None
      @ Out, meta, set(str), expected keys (empty if none)
    """
    # get keys as per base class
    keys = Model.provideExpectedMetaKeys(self)
    # add postprocessor keys
    try:
      keys = keys.union(self.interface.provideExpectedMetaKeys())
    except AttributeError:
      pass # either "interface" has no method for returning meta keys, or "interface" is not established yet.
    return keys

  def whatDoINeed(self):
    """
      This method is used mainly by the Simulation class at the Step construction stage.
      It is used for inquiring the class, which is implementing the method, about the kind of objects the class needs to
      be initialize. It is an abstract method -> It must be implemented in the derived class!
      NB. In this implementation, the method only calls the self.interface.whatDoINeed() method
      @ In, None
      @ Out, needDict, dict, dictionary of objects needed (class:tuple(object type{if None, Simulation does not check the type}, object name))
    """
    return self.interface.whatDoINeed()

  def generateAssembler(self,initDict):
    """
      This method is used mainly by the Simulation class at the Step construction stage.
      It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      NB. In this implementation, the method only calls the self.interface.generateAssembler(initDict) method
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    self.interface.generateAssembler(initDict)

  def _readMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Model._readMoreXML(self, xmlNode)
    self.interface = PostProcessors.returnInstance(self.subType,self)
    self.interface._readMoreXML(xmlNode)

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = Model.getInitParams(self)
    return paramDict

  def initialize(self,runInfo,inputs, initDict=None):
    """
      this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step
      after this call the next one will be run
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    self.workingDir               = os.path.join(runInfo['WorkingDir'],runInfo['stepName']) #generate current working dir
    self.interface.initialize(runInfo, inputs, initDict)
    self.mods = self.mods + list(set(utils.returnImportModuleString(inspect.getmodule(PostProcessors),True)) - set(self.mods))

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

    ## This may look a little weird, but due to how the parallel python library
    ## works, we are unable to pass a member function as a job because the
    ## pp library loses track of what self is, so instead we call it from the
    ## class and pass self in as the first parameter
    jobHandler.addJob((self, myInput, samplerType, kwargs), self.__class__.evaluateSample, str(0), modulesToImport=self.mods, forceUseThreads=True)

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
    returnValue = (Input, self.interface.run(Input))
    return returnValue

  def collectOutput(self,finishedjob,output,options=None):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ In, options, dict, optional, dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    self.interface.collectOutput(finishedjob,output)

  def createNewInput(self,myInput,samplerType,**kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      here only a PointSet is accepted a local copy of the values is performed.
      For a PointSet, only the last set of entries is copied
      The copied values are returned as a dictionary back
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, myInput, list, the inputs (list) to start from to generate the new one
    """
    return myInput
