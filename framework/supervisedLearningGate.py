"""
Created on December 6, 2016

@author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import inspect
import abc
import copy
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
import mathUtils
import utils
import SupervisedLearning
import MessageHandler
#Internal Modules End--------------------------------------------------------------------------------
class supervisedLearningGate(utils.metaclass_insert(abc.ABCMeta,BaseType),MessageHandler.MessageUser):
  """
    This class represents an interface with all the supervised learning algorithms
    It is a utility class needed to hide the discernment between time-dependent and static
    surrogate models
  """
  def __init__(self, ROMclass, messageHandler, **kwargs):
    """
      A constructor that will appropriately initialize a supervised learning object (static or time-dependent)
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, ROMclass, string, the surrogate model type
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    self.printTag                = 'SupervisedGate'
    self.messageHandler          = messageHandler
    self.initializationOptions   = kwargs
    self.amITrained              = False
    #the ROM is instanced and initialized
    # check how many targets
    if not 'Target' in self.initializationOptions.keys(): self.raiseAnError(IOError,'No Targets specified!!!')
    # return instance of the ROMclass
    modelInstance         = SupervisedLearning.returnInstance(ROMclass,self,**self.initializationOptions)
    # check if it is dynamic
    self.isADynamicModel  = modelInstance.isDynamic()
    # if it is dynamic and time series are passed in, self.SupervisedEngine is not going to be expanded, else it is going to
    self.SupervisedEngine = [modelInstance]
    # check if pivotParameter is specified and in case store it
    self.pivotParameterId = self.initializationOptions.pop("pivotParameter",'time')
    #
    self.historySteps = []

#     if 'SKLtype' in self.initializationOptions and 'MultiTask' in self.initializationOptions['SKLtype']:
#       self.initializationOptions['Target'] = targets
#       model = SupervisedLearning.returnInstance(self.subType,self,**self.initializationOptions)
#       for target in targets:
#         self.SupervisedEngine[target] = model
#     else:
#       for target in targets:
#         self.initializationOptions['Target'] = target
#         self.SupervisedEngine[target] =  SupervisedLearning.returnInstance(ROMclass,self,**self.initializationOptions)
#     # extend the list of modules this ROM depen on
#     self.mods = self.mods + list(set(utils.returnImportModuleString(inspect.getmodule(self.SupervisedEngine,True)) - set(self.mods)))
#     self.mods = self.mods + list(set(utils.returnImportModuleString(inspect.getmodule(SupervisedLearning),True)) - set(self.mods))


  def reset(self):
    """
      This method is aimed to reset the ROM
      @ In, None
      @ Out, None
    """
    for rom in self.SupervisedEngine: rom.reset()
    self.amITrained = False

  def getInitParams(self):
    """
      This function is called from the base class to print some of the information inside the class.
      Whatever is permanent in the class and not inherited from the parent class should be mentioned here
      The information is passed back in the dictionary. No information about values that change during the simulation are allowed
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = self.SupervisedEngine[-1].returnInitialParameters()
    return paramDict

  def train(self,trainingSet):
    """
      This function train the ROM this gate is linked to. This method is aimed to agnostically understand if a "time-dependent-like" ROM needs to be constructed.
      @ In, trainingSet, dict or list, data used to train the ROM; if a list is provided a temporal ROM is generated.
      @ Out, None
    """
    if type(trainingSet).__name__ not in  'dict': self.raiseAnError(IOError,"The training set is not a dictionary!")
    if len(trainingSet.keys()) == 0             : self.raiseAnError(IOError,"The training set is empty!")

    if any(type(x).__name__ == 'list' for x in trainingSet.values()):
      # we need to build a "time-dependent" ROM
      if self.pivotParameterId not in trainingSet.keys()            : self.raiseAnError(IOError,"the pivot parameter "+ self.pivotParameterId +" is not present in the training set. A time-dependent-like ROM cannot be created!")
      if type(trainingSet[self.pivotParameterId]).__name__ != 'list': self.raiseAnError(IOError,"the pivot parameter "+ self.pivotParameterId +" is not a list. Are you sure it is part of the output space of the training set?")
      self.historySteps = trainingSet.get(self.pivotParameterId)[-1]
      if len(self.historySteps) == 0: self.raiseAnError(IOError,"the training set is empty!")
      if self.isADynamicModel:
        # the ROM is able to manage the time dependency on its own
        self.SupervisedEngine[0].train(trainingSet)
      else:
        # we need to construct a chain of ROMs
        # the check on the number of time steps (consistency) is performed inside the historySnapShoots method
        # get the time slices
        newTrainingSet = mathUtils.historySnapShoots(trainingSet, len(self.historySteps))
        if type(newTrainingSet).__name__ != 'list': self.raiseAnError(IOError,newTrainingSet)
        # copy the original ROM
        originalROM = copy.deepcopy(self.SupervisedEngine[0])
        # start creating and training the time-dep ROMs
        self.SupervisedEngine = [] # [copy.deepcopy(originalROM) for _ in range(len(self.historySteps))]
        # train
        for ts in range(len(self.historySteps)):
          self.SupervisedEngine.append(copy.deepcopy(originalROM))
          self.SupervisedEngine[-1].train(newTrainingSet[ts])
    else:
      #self._replaceVariablesNamesWithAliasSystem(self.trainingSet, 'inout', False)
      self.SupervisedEngine[0].train(trainingSet)
    self.amITrained = True

  def confidence(self, request):
    """
      This is to get a value that is inversely proportional to the confidence that we have
      forecasting the target value for the given set of features. The reason to chose the inverse is because
      in case of normal distance this would be 1/distance that could be infinity
      @ In, request, dict, realizations request ({'feature1':np.array(n_realizations),'feature2',np.array(n_realizations)})
      @ Out, confidenceDict, dict, the dictionary where the confidence is stored for each target
    """
    if not self.amITrained: self.raiseAnError(RuntimeError, "ROM "+self.initializationOptions['name']+" has not been trained yet and, consequentially, can not be evaluated!")
    confidenceDict = {}
    for rom in self.SupervisedEngine:
      sliceEvaluation = rom.confidence(request)
      if len(confidenceDict.keys()) == 0:
        confidenceDict.update(sliceEvaluation)
      else:
        for key in confidenceDict.keys(): confidenceDict[key] = np.append(confidenceDict[key],sliceEvaluation[key])
    return confidenceDict

  def evaluate(self,request):
    """
      Method to perform the evaluation of a point or a set of points through the linked surrogate model
      @ In, request, dict, realizations request ({'feature1':np.array(n_realizations),'feature2',np.array(n_realizations)})
      @ Out, resultsDict, dict, dictionary of results ({target1:np.array,'target2':np.array}).
    """
    if not self.amITrained: self.raiseAnError(RuntimeError, "ROM "+self.initializationOptions['name']+" has not been trained yet and, consequentially, can not be evaluated!")
    resultsDict = {}
    for rom in self.SupervisedEngine:
      sliceEvaluation = rom.evaluate(request)
      if len(resultsDict.keys()) == 0:
        resultsDict.update(sliceEvaluation)
      else:
        for key in resultsDict.keys(): resultsDict[key] = np.append(resultsDict[key],sliceEvaluation[key])
    return resultsDict


__interfaceDict                         = {}
__interfaceDict['SupervisedGate'      ] = supervisedLearningGate
__base                                  = 'supervisedGate'

def returnInstance(gateType, ROMclass, caller, **kwargs):
  """
    This function return an instance of the request model type
    @ In, ROMclass, string, string representing the instance to create
    @ In, caller, instance, object that will share its messageHandler instance
    @ In, kwargs, dict, a dictionary specifying the keywords and values needed to create the instance.
    @ Out, returnInstance, instance, an instance of a ROM
  """
  try: return __interfaceDict[gateType](ROMclass, caller.messageHandler,**kwargs)
  except KeyError as ae: caller.raiseAnError(NameError,'not known '+__base+' type '+str(gateType))

def returnClass(ROMclass,caller):
  """
    This function return an instance of the request model type
    @ In, ROMclass, string, string representing the class to retrieve
    @ In, caller, instnace, object that will share its messageHandler instance
    @ Out, returnClass, the class definition of a ROM
  """
  try: return __interfaceDict[ROMclass]
  except KeyError: caller.raiseAnError(NameError,'not known '+__base+' type '+ROMclass)

