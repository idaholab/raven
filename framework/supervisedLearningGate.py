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
    #the ROM is instanced and initialized
    # check how many targets
    if not 'Target' in self.initializationOptions.keys(): self.raiseAnError(IOError,'No Targets specified!!!')
    #targets = self.initializationOptions['Target'].split(',')
    #self.howManyTargets = len(targets)
    # return instance of the ROMclass
    modelInstance         = SupervisedLearning.returnInstance(ROMclass,self,**self.initializationOptions)
    # check if it is dynamic
    self.isADynamicModel  = modelInstance.isDynamic()
    # if it is dynamic and time series are passed in, self.SupervisedEngine is not going to be expanded, else it is going to
    self.SupervisedEngine = [modelInstance]
    # check if pivotParameter is specified and in case store it
    self.pivotParameterId = self.initializationOptions.pop("pivotParameter",'time')

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
    pass

  def getInitParams(self):
    pass

  def train(self,trainingSet):

    if len(trainingSet.keys()) == 0: self.raiseAnError(IOError,"The training set is empty!")
    if type(trainingSet.values()[-1]).__name__ == 'list':
      # we need to build a "time-dependent" ROM
      if self.isADynamicModel:
        # the ROM is able to manage the time dependency on its own
        self.SupervisedEngine[0].train(trainingSet)
      else:
        # we need to construct a chain of ROMs
        pass
    else:
      pass


      if self.subType == 'ARMA':
        localInput = {}


        lupo = self._inputToInternal(trainingSet, full=True)
        aaaa = mathUtils.historySetWindow(trainingSet,2017)
        if type(trainingSet)!=dict:
          for entries in trainingSet.getParaKeys('inputs' ):
            if not trainingSet.isItEmpty(): localInput[entries] = copy.copy(np.array(trainingSet.getParam('input' ,1)[entries]))
            else:                      localInput[entries] = None
          for entries in trainingSet.getParaKeys('outputs'):
            if not trainingSet.isItEmpty(): localInput[entries] = copy.copy(np.array(trainingSet.getParam('output',1)[entries]))
            else:                      localInput[entries] = None
        self.trainingSet = copy.copy(localInput)
        if type(self.trainingSet) is dict:
          self.amITrained = True
          for instrom in self.SupervisedEngine.values():
            instrom.pivotParameter = np.asarray(trainingSet.getParam('output',1)[instrom.pivotParameterID])
            instrom.train(self.trainingSet)
            self.amITrained = self.amITrained and instrom.amITrained
          self.raiseADebug('add self.amITrained to currentParamters','FIXME')

      elif 'HistorySet' in type(trainingSet).__name__:
        #get the pivot parameter if specified
        self.historyPivotParameter = trainingSet._dataParameters.get('pivotParameter','time')
        #get the list of history steps if specified
        self.historySteps = trainingSet.getParametersValues('outputs').values()[0].get(self.historyPivotParameter,[])
        #store originals for future copying
        origRomCopies = {}
        for target,engine in self.SupervisedEngine.items():
          origRomCopies[target] = copy.deepcopy(engine)
        #clear engines for time-based storage
        self.SupervisedEngine = []
        outKeys = trainingSet.getParaKeys('outputs')
        targets = origRomCopies.keys()
        # check that all histories have the same length
        tmp = trainingSet.getParametersValues('outputs')
        for t in tmp:
          if t==1:
            self.numberOfTimeStep = len(tmp[t][outKeys[0]])
          else:
            if self.numberOfTimeStep != len(tmp[t][outKeys[0]]):
              self.raiseAnError(IOError,'DataObject can not be used to train a ROM: length of HistorySet is not consistent')
        # train the ROM
        self.trainingSet = mathUtils.historySetWindow(trainingSet,self.numberOfTimeStep)
        for ts in range(self.numberOfTimeStep):
          newRom = {}
          for target in targets:
            newRom[target] =  copy.deepcopy(origRomCopies[target])
          for target,instrom in newRom.items():
            # train the ROM
            self._replaceVariablesNamesWithAliasSystem(self.trainingSet[ts], 'inout', False)
            instrom.train(self.trainingSet[ts])
            self.amITrained = self.amITrained and instrom.amITrained
          self.SupervisedEngine.append(newRom)
        self.amITrained = True
      else:
        self.trainingSet = copy.copy(self._inputToInternal(trainingSet,full=True))
        if type(self.trainingSet) is dict:
          self._replaceVariablesNamesWithAliasSystem(self.trainingSet, 'inout', False)
          self.amITrained = True
          for instrom in self.SupervisedEngine.values():
            instrom.train(self.trainingSet)
            self.amITrained = self.amITrained and instrom.amITrained
          self.raiseADebug('add self.amITrained to currentParamters','FIXME')

  def confidence(self,request,target = None):
    pass

  def evaluate(self,request, target = None, timeInst = None):
    pass

  def run(self,Input):
    pass

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

