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
Created on December 6, 2016

@author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import inspect
import abc
import copy
import collections
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseInterface, MessageUser
from utils import mathUtils
from utils import utils
import SupervisedLearning
from EntityFactoryBase import EntityFactory
#Internal Modules End--------------------------------------------------------------------------------

#
#
#
#
class supervisedLearningGate(utils.metaclass_insert(abc.ABCMeta, BaseInterface), MessageUser):
  """
    This class represents an interface with all the supervised learning algorithms
    It is a utility class needed to hide the discernment between time-dependent and static
    surrogate models
  """
  def __init__(self, ROMclass, **kwargs):
    """
      A constructor that will appropriately initialize a supervised learning object (static or time-dependent)
      @ In, ROMclass, string, the surrogate model type
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    super().__init__()
    self.printTag = 'SupervisedGate'
    self.initializationOptions = kwargs
    self.amITrained = False
    self.ROMclass = ROMclass
    # members for clustered roms
    ### OLD ###
    #self._usingRomClustering = False    # are we using ROM clustering?
    #self._romClusterDivisions = {}      # which parameters do we cluster, and how are they subdivided?
    #self._romClusterLengths = {}        # OR which parameters do we cluster, and how long should each be?
    #self._romClusterMetrics = None      # list of requested metrics to apply (defaults to everything)
    #self._romClusterInfo = {}           # data that should persist across methods
    #self._romClusterPivotShift = None   # whether and how to normalize/shift subspaces
    #self._romClusterMap = None          # maps labels to the ROMs that are represented by it
    #self._romClusterFeatureTemplate = '{target}|{metric}|{id}' # standardized for consistency

    #the ROM is instanced and initialized
    #if ROM comes from a pickled rom, this gate is just a placeholder and the Targets check doesn't apply
    self.pickled = self.initializationOptions.pop('pickled', False)
    # check if pivotParameter is specified and in case store it
    self.pivotParameterId = self.initializationOptions.get("pivotParameter", 'time')
    # return instance of the ROMclass
    modelInstance = SupervisedLearning.factory.returnInstance(ROMclass, **self.initializationOptions)
    # check if the model can autonomously handle the time-dependency
    # (if not and time-dep data are passed in, a list of ROMs are constructed)
    self.canHandleDynamicData = modelInstance.isDynamic()
    # is this ROM  time-dependent ?
    self.isADynamicModel = False
    # if it is dynamic and time series are passed in, self.supervisedContainer is not going to be expanded, else it is going to
    self.supervisedContainer = [modelInstance]
    self.historySteps = []

    nameToClass = {'segment': 'Segments',
                   'cluster': 'Clusters',
                   'interpolate': 'Interpolated'}
    ### ROMCollection ###
    # if the ROM targeted by this gate is a cluster, create the cluster now!
    if 'Segment' in self.initializationOptions:
      # read from specs directly
      segSpecs = self.initializationOptions['paramInput'].findFirst('Segment')
      # determine type of segment to load -> limited by InputData to specific options
      segType = segSpecs.parameterValues.get('grouping', 'segment')
      self.initializationOptions['modelInstance'] = modelInstance
      SVL = SupervisedLearning.factory.returnInstance(nameToClass[segType], **self.initializationOptions)
      self.supervisedContainer = [SVL]

  def __getstate__(self):
    """
      This function return the state of the ROM
      @ In, None
      @ Out, state, dict, it contains all the information needed by the ROM to be initialized
    """
    # clear input specs, as they should all be read in by now
    ## this isn't a great implementation; we should make paramInput picklable instead!
    self.initializationOptions.pop('paramInput', None)
    for eng in self.supervisedContainer:
      eng.initOptionDict.pop('paramInput',None)
    # capture what is normally pickled
    state = self.__dict__.copy()
    if not self.amITrained:
      supervisedEngineObj = state.pop("supervisedContainer")
      del supervisedEngineObj
    return state

  def __setstate__(self, newstate):
    """
      Initialize the ROM with the data contained in newstate
      @ In, newstate, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    self.__dict__.update(newstate)
    if not newstate['amITrained']:
      # NOTE this will fail if the ROM requires the paramInput spec! Fortunately, you shouldn't pickle untrained.
      modelInstance = SupervisedLearning.factory.returnInstance(self.ROMclass, **self.initializationOptions)
      self.supervisedContainer  = [modelInstance]

  def setAdditionalParams(self, params):
    """
      Sets parameters aside from initialization, such as during deserialization.
      @ In, params, dict, parameters to set (dependent on ROM)
      @ Out, None
    """
    for rom in self.supervisedContainer:
      rom.setAdditionalParams(params)

  def reset(self):
    """
      This method is aimed to reset the ROM
      @ In, None
      @ Out, None
    """
    for rom in self.supervisedContainer:
      rom.reset()
    self.amITrained = False

  def reseed(self,seed):
    """
      Used to reset the seed of the underlying ROMs.
      @ In, seed, int, new seed to use
      @ Out, None
    """
    for rom in self.supervisedContainer:
      rom.reseed(seed)

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = self.supervisedContainer[-1].returnInitialParameters()
    return paramDict

  def provideExpectedMetaKeys(self):
    """
      Overrides the base class method to assure child engine is also polled for its keys.
      @ In, None
      @ Out, meta, tuple, (set(str),dict), expected keys (empty if none) and the indexes related to expected keys
    """
    # get from engine
    keys, params = self.supervisedContainer[0].provideExpectedMetaKeys()
    return keys, params

  def train(self, trainingSet, assembledObjects=None):
    """
      This function train the ROM this gate is linked to. This method is aimed to agnostically understand if a "time-dependent-like" ROM needs to be constructed.
      @ In, trainingSet, dict or list, data used to train the ROM; if a list is provided a temporal ROM is generated.
      @ In, assembledObjects, dict, optional, objects that the ROM Model has assembled via the Assembler
      @ Out, None
    """
    if type(trainingSet).__name__ not in  'dict':
      self.raiseAnError(IOError, "The training set is not a dictionary!")
    if not list(trainingSet.keys()):
      self.raiseAnError(IOError, "The training set is empty!")

    # provide assembled objects to supervised container
    if assembledObjects is None:
      assembledObjects = {}

    self.supervisedContainer[0].setAssembledObjects(assembledObjects)

    # if training using ROMCollection, special treatment
    if isinstance(self.supervisedContainer[0], SupervisedLearning.Collection):
      self.supervisedContainer[0].train(trainingSet)
    else:
      # not a collection # TODO move time-dependent snapshots to collection!
      ## time-dependent or static ROM?
      if any(type(x).__name__ == 'list' for x in trainingSet.values()):
        # we need to build a "time-dependent" ROM
        self.isADynamicModel = True
        if self.pivotParameterId not in list(trainingSet.keys()):
          self.raiseAnError(IOError, 'The pivot parameter "{}" is not present in the training set.'.format(self.pivotParameterId),
                            'A time-dependent-like ROM cannot be created!')
        if type(trainingSet[self.pivotParameterId]).__name__ != 'list':
          self.raiseAnError(IOError, 'The pivot parameter "{}" is not a list.'.format(self.pivotParameterId),
                            " Are you sure it is part of the output space of the training set?")
        self.historySteps = trainingSet.get(self.pivotParameterId)[-1]
        if not len(self.historySteps):
          self.raiseAnError(IOError, "the training set is empty!")
        # intrinsically time-dependent or does the Gate need to handle it?
        if self.canHandleDynamicData:
          # the ROM is able to manage the time dependency on its own
          self.supervisedContainer[0].train(trainingSet)
        else:
          # TODO we can probably migrate this time-dependent handling to a type of ROMCollection!
          # we need to construct a chain of ROMs
          # the check on the number of time steps (consistency) is performed inside the historySnapShoots method
          # get the time slices
          newTrainingSet = mathUtils.historySnapShoots(trainingSet, len(self.historySteps))
          assert type(newTrainingSet).__name__ == 'list'
          # copy the original ROM
          originalROM = self.supervisedContainer[0]
          # start creating and training the time-dep ROMs
          self.supervisedContainer = [] # [copy.deepcopy(originalROM) for _ in range(len(self.historySteps))]
          # train
          for ts in range(len(self.historySteps)):
            self.supervisedContainer.append(copy.deepcopy(originalROM))
            self.supervisedContainer[-1].train(newTrainingSet[ts])
      # if a static ROM ...
      else:
        #self._replaceVariablesNamesWithAliasSystem(self.trainingSet, 'inout', False)
        self.supervisedContainer[0].train(trainingSet)
    # END if ROMCollection
    self.amITrained = True

  def confidence(self, request):
    """
      This is to get a value that is inversely proportional to the confidence that we have
      forecasting the target value for the given set of features. The reason to chose the inverse is because
      in case of normal distance this would be 1/distance that could be infinity
      @ In, request, dict, realizations request ({'feature1':np.array(n_realizations),'feature2',np.array(n_realizations)})
      @ Out, confidenceDict, dict, the dictionary where the confidence is stored for each target
    """
    if not self.amITrained:
      self.raiseAnError(RuntimeError, "ROM "+self.initializationOptions['name']+" has not been trained yet and, consequentially, can not be evaluated!")
    confidenceDict = {}
    for rom in self.supervisedContainer:
      sliceEvaluation = rom.confidence(request)
      if len(list(confidenceDict.keys())) == 0:
        confidenceDict.update(sliceEvaluation)
      else:
        for key in confidenceDict.keys():
          confidenceDict[key] = np.append(confidenceDict[key],sliceEvaluation[key])
    return confidenceDict

  # compatibility with BaseInterface requires having a "run" method
  # TODO during SVL rework, "run" should probably replace "evaluate", maybe?
  def run(self, request):
    """
      Method to perform the evaluation of a point or a set of points through the previous trained supervisedLearning algorithm
      NB.the supervisedLearning object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires.
      @ In, request, dict, realizations request ({'feature1':np.array(n_realizations),
                                                  'feature2',np.array(n_realizations)})
      @ Out, run, dict, dictionary of results ({target1:np.array,'target2':np.array}).
    """
    return self.evaluate(request)

  def evaluate(self,request):
    """
      Method to perform the evaluation of a point or a set of points through the linked surrogate model
      @ In, request, dict, realizations request ({'feature1':np.array(n_realizations),
                                                  'feature2',np.array(n_realizations)})
      @ Out, resultsDict, dict, dictionary of results ({target1:np.array,'target2':np.array}).
    """
    if self.pickled:
      self.raiseAnError(RuntimeError,'ROM "'+self.initializationOptions['name']+'" has not been loaded yet!  Use an IOStep to load it.')
    if not self.amITrained:
      self.raiseAnError(RuntimeError, "ROM "+self.initializationOptions['name']+" has not been trained yet and, consequentially, can not be evaluated!")
    resultsDict = {}
    if isinstance(self.supervisedContainer[0], SupervisedLearning.Collection):
      resultsDict = self.supervisedContainer[0].evaluate(request)
    else:
      for rom in self.supervisedContainer:
        sliceEvaluation = rom.evaluate(request)
        if len(list(resultsDict.keys())) == 0:
          resultsDict.update(sliceEvaluation)
        else:
          for key in resultsDict.keys():
            resultsDict[key] = np.append(resultsDict[key],sliceEvaluation[key])
    return resultsDict

class LearningGateFactory(EntityFactory):
  """
    Specific factory for LearningGate
  """
  def returnInstance(self, Type, romClass, **kwargs):
    """
      Return an instance of the requested type
      @ In, Type, str, string name of gate requested
      @ In, romClass, str, string representing the instance to create
      @ In, kwargs, dict, a dictionary specifying the keywords and values needed to create the instance.
      @ Out, returnInstance, instance, an instance of a ROM
    """
    cls = self.returnClass(Type)
    instance = cls(romClass, **kwargs)
    return instance

factory = LearningGateFactory('supervisedGate')
factory.registerType('SupervisedGate', supervisedLearningGate)
