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
  Created on July 10, 2013

  Module containing all supported type of ROM aka Surrogate Models etc
  here we intend ROM as super-visioned learning,
  where we try to understand the underlying model by a set of labeled sample
  a sample is composed by (feature,label) that is easy translated in (input,output)
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
from numpy import average
from crow_modules.distribution1Dpy2 import CDF
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
from sklearn import linear_model
from sklearn import svm
from sklearn import multiclass
from sklearn import naive_bayes
from sklearn import neighbors

from sklearn import qda
from sklearn import lda
# from sklearn import discriminant_analysis

from sklearn import tree

from sklearn import gaussian_process
import numpy as np
import abc
import ast
from operator import itemgetter
from collections import OrderedDict
from scipy import spatial
from scipy import optimize
from sklearn.neighbors.kde import KernelDensity
import math
import copy
import itertools
from scipy import spatial
from collections import OrderedDict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
from utils import mathUtils
import sys
import MessageHandler
import Distributions
interpolationND = utils.find_interpolationND()
#Internal Modules End--------------------------------------------------------------------------------

class superVisedLearning(utils.metaclass_insert(abc.ABCMeta),MessageHandler.MessageUser):
  """
    This is the general interface to any superVisedLearning learning method.
    Essentially it contains a train method and an evaluate method
  """
  returnType       = ''    # this describe the type of information generated the possibility are 'boolean', 'integer', 'float'
  qualityEstType   = []    # this describe the type of estimator returned known type are 'distance', 'probability'. The values are returned by the self.__confidenceLocal__(Features)
  ROMtype          = ''    # the broad class of the interpolator
  ROMmultiTarget   = False #
  ROMtimeDependent = False # is this ROM able to treat time-like (any monotonic variable) explicitly in its formulation?

  @staticmethod
  def checkArrayConsistency(arrayIn):
    """
      This method checks the consistency of the in-array
      @ In, arrayIn, object,  It should be an array
      @ Out, (consistent, 'error msg'), tuple, tuple[0] is a bool (True -> everything is ok, False -> something wrong), tuple[1], string ,the error mesg
    """
    #checking if None provides a more clear message about the problem
    if arrayIn is None: return (False,' The object is None, and contains no entries!')
    if type(arrayIn).__name__ == 'list':
      if self.isDynamic():
        for cnt, elementArray in enumerate(arrayIn):
          resp = checkArrayConsistency(elementArray)
          if not resp[0]: return (False,' The element number '+str(cnt)+' is not a consistent array. Error: '+resp[1])
      else:
        return (False,' The list type is allowed for dynamic ROMs only')
    else:
      if type(arrayIn).__name__ not in ['ndarray','c1darray']: return (False,' The object is not a numpy array. Got type: '+type(arrayIn).__name__)
      if len(np.asarray(arrayIn).shape) > 1                  : return(False, ' The array must be 1-d. Got shape: '+str(np.asarray(arrayIn).shape))
    return (True,'')

  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately initialize a supervised learning object
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    self.printTag          = 'Supervised'
    self.messageHandler    = messageHandler
    self._dynamicHandling = False
    #booleanFlag that controls the normalization procedure. If true, the normalization is performed. Default = True
    if kwargs != None: self.initOptionDict = kwargs
    else             : self.initOptionDict = {}
    if 'Features' not in self.initOptionDict.keys(): self.raiseAnError(IOError,'Feature names not provided')
    if 'Target'   not in self.initOptionDict.keys(): self.raiseAnError(IOError,'Target name not provided')
    self.features = self.initOptionDict['Features'].split(',')
    self.target   = self.initOptionDict['Target'  ].split(',')
    self.initOptionDict.pop('Target')
    self.initOptionDict.pop('Features')
    self.verbosity = self.initOptionDict['verbosity'] if 'verbosity' in self.initOptionDict else None
    for target in self.target:
      if self.features.count(target) > 0: self.raiseAnError(IOError,'The target "'+target+'" is also in the feature space!')
    #average value and sigma are used for normalization of the feature data
    #a dictionary where for each feature a tuple (average value, sigma)
    self.muAndSigmaFeatures = {}
    #these need to be declared in the child classes!!!!
    self.amITrained         = False

  def initialize(self,idict):
    """
      Initialization method
      @ In, idict, dict, dictionary of initialization parameters
      @ Out, None
    """
    pass #Overloaded by (at least) GaussPolynomialRom

  def train(self,tdict):
    """
      Method to perform the training of the superVisedLearning algorithm
      NB.the superVisedLearning object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires. So far the base class will do the translation into numpy
      @ In, tdict, dict, training dictionary
      @ Out, None
    """
    if type(tdict) != dict:
      self.raiseAnError(TypeError,'In method "train", the training set needs to be provided through a dictionary. Type of the in-object is ' + str(type(tdict)))
    names, values  = list(tdict.keys()), list(tdict.values())
    ## This is for handling the special case needed by SKLtype=*MultiTask* that
    ## requires multiple targets.

    targetValues = []
    for target in self.target:
      if target in names: targetValues.append(values[names.index(target)])
      else              : self.raiseAnError(IOError,'The target '+target+' is not in the training set')

    #FIXME: when we do not support anymore numpy <1.10, remove this IF STATEMENT
    if int(np.__version__.split('.')[1]) >= 10:
      targetValues = np.stack(targetValues, axis=-1)
    else:
      sl = (slice(None),) * np.asarray(targetValues[0]).ndim + (np.newaxis,)
      targetValues = np.concatenate([np.asarray(arr)[sl] for arr in targetValues], axis=np.asarray(targetValues[0]).ndim)
    # construct the evaluation matrixes
    featureValues = np.zeros(shape=(len(targetValues),len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names:
        self.raiseAnError(IOError,'The feature sought '+feat+' is not in the training set')
      else:
        valueToUse = values[names.index(feat)]
        resp = self.checkArrayConsistency(valueToUse)
        if not resp[0]: self.raiseAnError(IOError,'In training set for feature '+feat+':'+resp[1])
        valueToUse = np.asarray(valueToUse)
        if valueToUse.size != featureValues[:,0].size:
          self.raiseAWarning('feature values:',featureValues[:,0].size,tag='ERROR')
          self.raiseAWarning('target values:',valueToUse.size,tag='ERROR')
          self.raiseAnError(IOError,'In training set, the number of values provided for feature '+feat+' are != number of target outcomes!')
        self._localNormalizeData(values,names,feat)
        featureValues[:,cnt] = (valueToUse - self.muAndSigmaFeatures[feat][0])/self.muAndSigmaFeatures[feat][1]
    self.__trainLocal__(featureValues,targetValues)
    self.amITrained = True

  def _localNormalizeData(self,values,names,feat):
    """
      Method to normalize data based on the mean and standard deviation.  If undesired for a particular ROM,
      this method can be overloaded to simply pass (see, e.g., GaussPolynomialRom).
      @ In, values, list, list of feature values (from tdict)
      @ In, names, list, names of features (from tdict)
      @ In, feat, list, list of features (from ROM)
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = mathUtils.normalizationFactors(values[names.index(feat)])

  def confidence(self,edict):
    """
      This call is used to get an estimate of the confidence in the prediction.
      The base class self.confidence will translate a dictionary into numpy array, then call the local confidence
      @ In, edict, dict, evaluation dictionary
      @ Out, confidence, float, the confidence
    """
    if type(edict) != dict: self.raiseAnError(IOError,'method "confidence". The inquiring set needs to be provided through a dictionary. Type of the in-object is ' + str(type(edict)))
    names, values   = list(edict.keys()), list(edict.values())
    for index in range(len(values)):
      resp = self.checkArrayConsistency(values[index])
      if not resp[0]: self.raiseAnError(IOError,'In evaluate request for feature '+names[index]+':'+resp[1])
    featureValues = np.zeros(shape=(values[0].size,len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names: self.raiseAnError(IOError,'The feature sought '+feat+' is not in the evaluate set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]: self.raiseAnError(IOError,'In training set for feature '+feat+':'+resp[1])
        featureValues[:,cnt] = values[names.index(feat)]
    return self.__confidenceLocal__(featureValues)

  def evaluate(self,edict):
    """
      Method to perform the evaluation of a point or a set of points through the previous trained superVisedLearning algorithm
      NB.the superVisedLearning object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires.
      @ In, edict, dict, evaluation dictionary
      @ Out, evaluate, numpy.array, evaluated points
    """
    if type(edict) != dict: self.raiseAnError(IOError,'method "evaluate". The evaluate request/s need/s to be provided through a dictionary. Type of the in-object is ' + str(type(edict)))
    names, values  = list(edict.keys()), list(edict.values())
    for index in range(len(values)):
      resp = self.checkArrayConsistency(values[index])
      if not resp[0]: self.raiseAnError(IOError,'In evaluate request for feature '+names[index]+':'+resp[1])
    # construct the evaluation matrix
    featureValues = np.zeros(shape=(values[0].size,len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names: self.raiseAnError(IOError,'The feature sought '+feat+' is not in the evaluate set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]: self.raiseAnError(IOError,'In training set for feature '+feat+':'+resp[1])
        featureValues[:,cnt] = ((values[names.index(feat)] - self.muAndSigmaFeatures[feat][0]))/self.muAndSigmaFeatures[feat][1]
    return self.__evaluateLocal__(featureValues)

  def reset(self):
    """
      Reset ROM
    """
    self.amITrained = False
    self.__resetLocal__()

  def returnInitialParameters(self):
    """
      override this method to return the fix set of parameters of the ROM
      @ In, None
      @ Out, iniParDict, dict, initial parameter dictionary
    """
    iniParDict = dict(list(self.initOptionDict.items()) + list({'returnType':self.__class__.returnType,'qualityEstType':self.__class__.qualityEstType,'Features':self.features,
                                             'Target':self.target,'returnType':self.__class__.returnType}.items()) + list(self.__returnInitialParametersLocal__().items()))
    return iniParDict

  def returnCurrentSetting(self):
    """
      return the set of parameters of the ROM that can change during simulation
      @ In, None
      @ Out, currParDict, dict, current parameter dictionary
    """
    currParDict = dict({'Trained':self.amITrained}.items() + self.__CurrentSettingDictLocal__().items())
    return currParDict

  def printXML(self,outFile,pivotVal,options={}):
    """
      Allows the SVE to put whatever it wants into an XML to print to file.
      @ In, outFile, Files.File, either StaticXMLOutput or DynamicXMLOutput file
      @ In, pivotVal, float, value of pivot parameters to use in printing if dynamic
      @ In, options, dict, optional, dict of string-based options to use, including filename, things to print, etc
      @ Out, None
    """
    self._localPrintXML(outFile,pivotVal,options)

  def _localPrintXML(self,node,options=None):
    """
      Specific local method for printing anything desired to xml file.  Overwrite in inheriting classes.
      @ In, node, the node to which strings should have text added
      @ In, options, dict of string-based options to use, including filename, things to print, etc
      @ Out, None
    """
    node.addText('ROM of type '+str(self.printTag.strip())+' has no special output options.')

  def isDynamic(self):
    """
      This method is a utility function that tells if the relative ROM is able to
      treat dynamic data (e.g. time-series) on its own or not (Primarly called by LearningGate)
      @ In, None
      @ Out, isDynamic, bool, True if the ROM is able to treat dynamic data, False otherwise
    """
    return self._dynamicHandling

  @abc.abstractmethod
  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on samples in featureVals with responses y.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature values
      @ Out, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """

  @abc.abstractmethod
  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      This could be distance or probability or anything else, the type needs to be declared in the variable cls.qualityEstType
      @ In, featureVals, 2-D numpy array , [n_samples,n_features]
      @ Out, __confidenceLocal__, float, the confidence
    """

  @abc.abstractmethod
  def __evaluateLocal__(self,featureVals):
    """
      @ In,  featureVals, np.array, 2-D numpy array [n_samples,n_features]
      @ Out, targetVals , np.array, 1-D numpy array [n_samples]
    """

  @abc.abstractmethod
  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """

  @abc.abstractmethod
  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """

  @abc.abstractmethod
  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """#
#
#
class NDinterpolatorRom(superVisedLearning):
  """
  A Reduced Order Model for interpolating N-dimensional data
  """
  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler, MessageHandler, a MessageHandler object in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
      @ Out, None
    """
    superVisedLearning.__init__(self,messageHandler,**kwargs)
    self.interpolator = []    # pointer to the C++ (crow) interpolator (list of targets)
    self.featv        = None  # list of feature variables
    self.targv        = None  # list of target variables
    self.printTag = 'ND Interpolation ROM'

  def __getstate__(self):
    """
      Overwrite state (for pickle-ing)
      we do not pickle the HDF5 (C++) instance
      but only the info to re-load it
      @ In, None
      @ Out, state, dict, namespace dictionary
    """
    # capture what is normally pickled
    state = self.__dict__.copy()
    if 'interpolator' in state.keys():
      a = state.pop("interpolator")
      del a
    return state

  def __setstate__(self, newstate):
    """
      Initialize the ROM with the data contained in newstate
      @ In, newstate, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    self.__dict__.update(newstate)
    self.__initLocal__()
    #only train if the original copy was trained
    if self.amITrained:
      self.__trainLocal__(self.featv,self.targv)

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on samples in featureVals with responses y.
      For an one-class model, +1 or -1 is returned.

      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature values
      @ Out, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """
    self.featv, self.targv = featureVals,targetVals
    featv = interpolationND.vectd2d(featureVals[:][:])
    for index, target in enumerate(self.target):
      targv = interpolationND.vectd(targetVals[:,index])
      self.interpolator[index].fit(featv,targv)

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals, 2-D numpy array, [n_samples,n_features]
      @ Out, confidence, float, the confidence
    """
    self.raiseAnError(NotImplementedError,'NDinterpRom   : __confidenceLocal__ method must be implemented!')

  def __evaluateLocal__(self,featureVals):
    """
      Perform regression on samples in featureVals.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, numpy.array 2-D, features
      @ Out, prediction, numpy.array 1-D, predicted values
    """
    prediction = {} #np.zeros((featureVals.shape[0]))
    for index, target in enumerate(self.target):
      prediction[target] = np.zeros((featureVals.shape[0]))
      for n_sample in range(featureVals.shape[0]):
        featv = interpolationND.vectd(featureVals[n_sample][:])
        prediction[target][n_sample] = self.interpolator[index].interpolateAt(featv)
      self.raiseAMessage('NDinterpRom   : Prediction by ' + self.__class__.ROMtype + ' for target '+target+'. Predicted value is ' + str(prediction[target][n_sample]))
    return prediction

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    params = {}
    return params

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    self.raiseAnError(NotImplementedError,'NDinterpRom   : __returnCurrentSettingLocal__ method must be implemented!')
#
#
#
#
class GaussPolynomialRom(superVisedLearning):
  """
    Gauss Polynomial Rom Class
  """
  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals, 2-D numpy array, [n_samples,n_features]
      @ Out, confidence, float, the confidence
    """
    pass

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    pass

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    pass

  def __initLocal__(self):
    """
      Method used to add additional initialization features used by pickling
      @ In, None
      @ Out, None
    """
    pass

  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    superVisedLearning.__init__(self,messageHandler,**kwargs)
    self.initialized   = False #only True once self.initialize has been called
    self.interpolator  = None #FIXME what's this?
    self.printTag      = 'GAUSSgpcROM('+'-'.join(self.target)+')'
    self.indexSetType  = None #string of index set type, TensorProduct or TotalDegree or HyperbolicCross
    self.indexSetVals  = []   #list of tuples, custom index set to use if CustomSet is the index set type
    self.maxPolyOrder  = None #integer of relative maximum polynomial order to use in any one dimension
    self.itpDict       = {}   #dict of quad,poly,weight choices keyed on varName
    self.norm          = None #combined distribution normalization factors (product)
    self.sparseGrid    = None #Quadratures.SparseGrid object, has points and weights
    self.distDict      = None #dict{varName: Distribution object}, has point conversion methods based on quadrature
    self.quads         = None #dict{varName: Quadrature object}, has keys for distribution's point conversion methods
    self.polys         = None #dict{varName: OrthoPolynomial object}, has polynomials for evaluation
    self.indexSet      = None #array of tuples, polynomial order combinations
    self.polyCoeffDict = None #dict{index set point, float}, polynomial combination coefficients for each combination
    self.numRuns       = None #number of runs to generate ROM; default is len(self.sparseGrid)
    self.itpDict       = {}   #dict{varName: dict{attribName:value} }
    self.featv         = None  # list of feature variables
    self.targv         = None  # list of target variables
    self.mean          = None
    self.variance      = None
    self.sdx           = None
    self.partialVariances = None
    self.sparseGridType    = 'smolyak' #type of sparse quadrature to use,default smolyak
    self.sparseQuadOptions = ['smolyak','tensor'] # choice of sparse quadrature construction methods

    for key,val in kwargs.items():
      if key=='IndexSet':self.indexSetType = val
      elif key=='IndexPoints':
        self.indexSetVals=[]
        strIndexPoints = val.strip()
        strIndexPoints = strIndexPoints.replace(' ','').replace('\n','').strip('()')
        strIndexPoints = strIndexPoints.split('),(')
        self.raiseADebug(strIndexPoints)
        for s in strIndexPoints:
          self.indexSetVals.append(tuple(int(i) for i in s.split(',')))
        self.raiseADebug('points',self.indexSetVals)
      elif key=='PolynomialOrder': self.maxPolyOrder = val
      elif key=='Interpolation':
        for var,val in val.items():
          self.itpDict[var]={'poly'  :'DEFAULT',
                             'quad'  :'DEFAULT',
                             'weight':'1'}
          for atrName,atrVal in val.items():
            if atrName in ['poly','quad','weight']: self.itpDict[var][atrName]=atrVal
            else: self.raiseAnError(IOError,'Unrecognized option: '+atrName)
      elif key == 'SparseGrid':
        if val.lower() not in self.sparseQuadOptions:
          self.raiseAnError(IOError,'No such sparse quadrature implemented: %s.  Options are %s.' %(val,str(self.sparseQuadOptions)))
        self.sparseGridType = val

    if not self.indexSetType:
      self.raiseAnError(IOError,'No IndexSet specified!')
    if self.indexSetType=='Custom':
      if len(self.indexSetVals)<1: self.raiseAnError(IOError,'If using CustomSet, must specify points in <IndexPoints> node!')
      else:
        for i in self.indexSetVals:
          if len(i)<len(self.features): self.raiseAnError(IOError,'CustomSet points',i,'is too small!')
    if not self.maxPolyOrder:
      self.raiseAnError(IOError,'No maxPolyOrder specified!')
    if self.maxPolyOrder < 1:
      self.raiseAnError(IOError,'Polynomial order cannot be less than 1 currently.')

  def _localPrintXML(self,outFile,pivotVal,options={}):
    """
      Adds requested entries to XML node.
      @ In, outFile, Files.File, either StaticXMLOutput or DynamicXMLOutput file
      @ In, pivotVal, float, value of pivot parameters to use in printing if dynamic
      @ In, options, dict, optional, dict of string-based options to use, including filename, things to print, etc
        May include:
        'what': comma-separated string list, the qualities to print out
        'pivotVal': float value of dynamic pivotParam value
      @ Out, None
    """
    if not self.amITrained: self.raiseAnError(RuntimeError,'ROM is not yet trained!')
    #reset stats so they're fresh for this calculation
    self.mean=None
    sobolIndices = None
    partialVars = None
    sobolTotals = None
    variance = None
    #establish what we can handle, and how
    scalars = ['mean','expectedValue','variance','samples']
    vectors = ['polyCoeffs','partialVariance','sobolIndices','sobolTotalIndices']
    canDo = scalars + vectors
    #lowercase for convenience
    scalars = list(s.lower() for s in scalars)
    vectors = list(v.lower() for v in vectors)
    #establish requests, defaulting to "all"
    if 'what' in options.keys():
      requests = list(o.strip() for o in options['what'].split(','))
    else:
      requests =['all']
    # Target
    target = options.get('Target',self.target[0])
    #handle "all" option
    if 'all' in requests:
      requests = canDo
    # loop over the requested items
    for request in requests:
      request=request.strip()
      if request.lower() in scalars:
        if request.lower() in ['mean','expectedvalue']:
          #only calculate the mean once per printing
          if self.mean is None:
            self.mean = self.__mean__(target)
          val = self.mean
        elif request.lower() == 'variance':
          if variance is None:
            variance = self.__variance__(target)
          val = variance
        elif request.lower() == 'samples':
          if self.numRuns!=None:
            val = self.numRuns
          else:
            val = len(self.sparseGrid)
        outFile.addScalar(target,request,val,pivotVal=pivotVal)
      elif request.lower() in vectors:
        if request.lower() == 'polycoeffs':
          valueDict = OrderedDict()
          valueDict['inputVariables'] = ','.join(self.features)
          keys = self.polyCoeffDict[target].keys()
          keys.sort()
          for key in keys:
            valueDict['_'+'_'.join(str(k) for k in key)+'_'] = self.polyCoeffDict[target][key]
        elif request.lower() in ['partialvariance','sobolindices','soboltotalindices']:
          if sobolIndices is None or partialVars is None:
            sobolIndices,partialVars = self.getSensitivities(target)
          if sobolTotals is None:
            sobolTotals = self.getTotalSensitivities(sobolIndices)
          #sort by value
          entries = []
          if request.lower() in ['partialvariance','sobolindices']: #these both will have same sort
            for key in sobolIndices.keys():
              entries.append( ('.'.join(key),partialVars[key],key) )
          elif request.lower() in ['soboltotalindices']:
            for key in sobolTotals.keys():
              entries.append( ('.'.join(key),sobolTotals[key],key) )
          entries.sort(key=lambda x: abs(x[1]),reverse=True)
          #add entries to results list
          valueDict=OrderedDict()
          for entry in entries:
            name,_,key = entry
            if request.lower() == 'partialvariance':
              valueDict[name] = partialVars[key]
            elif request.lower() == 'sobolindices':
              valueDict[name] = sobolIndices[key]
            elif request.lower() == 'soboltotalindices':
              valueDict[name] = sobolTotals[key]
        outFile.addVector(target,request,valueDict,pivotVal=pivotVal)
      else:
        self.raiseAWarning('ROM does not know how to return "'+request+'".  Skipping...')

  def _localNormalizeData(self,values,names,feat):
    """
      Overwrites default normalization procedure.
      @ In, values, list(float), unused
      @ In, names, list(string), unused
      @ In, feat, string, feature to (not) normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def interpolationInfo(self):
    """
      Returns the interpolation information
      @ In, None
      @ Out, interpValues, dict, dictionary of interpolation information
    """
    interpValues = dict(self.itpDict)
    return interpValues

  def initialize(self,idict):
    """
      Initializes the instance.
      @ In, idict, dict, objects needed to initalize
      @ Out, None
    """
    self.sparseGrid     = idict.get('SG'        ,None)
    self.distDict       = idict.get('dists'     ,None)
    self.quads          = idict.get('quads'     ,None)
    self.polys          = idict.get('polys'     ,None)
    self.indexSet       = idict.get('iSet'      ,None)
    self.numRuns        = idict.get('numRuns'   ,None)
    #make sure requireds are not None
    if self.sparseGrid is None: self.raiseAnError(RuntimeError,'Tried to initialize without key object "SG"   ')
    if self.distDict   is None: self.raiseAnError(RuntimeError,'Tried to initialize without key object "dists"')
    if self.quads      is None: self.raiseAnError(RuntimeError,'Tried to initialize without key object "quads"')
    if self.polys      is None: self.raiseAnError(RuntimeError,'Tried to initialize without key object "polys"')
    if self.indexSet   is None: self.raiseAnError(RuntimeError,'Tried to initialize without key object "iSet" ')
    self.initialized = True

  def _multiDPolyBasisEval(self,orders,pts):
    """
      Evaluates each polynomial set at given orders and points, returns product.
      @ In orders, tuple(int), polynomial orders to evaluate
      @ In pts, tuple(float), values at which to evaluate polynomials
      @ Out, tot, float, product of polynomial evaluations
    """
    tot=1
    for i,(o,p) in enumerate(zip(orders,pts)):
      varName = self.sparseGrid.varNames[i]
      tot*=self.polys[varName](o,p)
    return tot

  def __trainLocal__(self,featureVals,targetVals):
    """
      Trains ROM.
      @ In, featureVals, np.ndarray, feature values
      @ In, targetVals, np.ndarray, target values
    """
    #check to make sure ROM was initialized
    if not self.initialized:
      self.raiseAnError(RuntimeError,'ROM has not yet been initialized!  Has the Sampler associated with this ROM been used?')
    self.raiseADebug('training',self.features,'->',self.target)
    self.featv, self.targv = featureVals,targetVals
    self.polyCoeffDict = {key: dict({}) for key in self.target}
    #check equality of point space
    self.raiseADebug('...checking required points are available...')
    fvs = []
    tvs = {key: list({}) for key in self.target}
    sgs = list(self.sparseGrid.points())
    missing=[]
    kdTree = spatial.KDTree(featureVals)
    #TODO this is slowest loop in this algorithm, by quite a bit.
    for pt in sgs:
      #KDtree way
      distances,idx = kdTree.query(pt,k=1,distance_upper_bound=1e-9) #FIXME how to set the tolerance generically?
      #KDTree repots a "not found" as at infinite distance with index len(data)
      if idx >= len(featureVals):
        found = False
      else:
        found = True
        point = tuple(featureVals[idx])
      #end KDTree way
      if found:
        fvs.append(point)
        for cnt, target in enumerate(self.target):  tvs[target].append(targetVals[idx,cnt])
      else:
        missing.append(pt)
    if len(missing)>0:
      msg='\n'
      msg+='DEBUG missing feature vals:\n'
      for i in missing:
        msg+='  '+str(i)+'\n'
      self.raiseADebug(msg)
      self.raiseADebug('sparse:',sgs)
      self.raiseADebug('solns :',fvs)
      self.raiseAnError(IOError,'input values do not match required values!')
    #make translation matrix between lists, also actual-to-standardized point map
    self.raiseADebug('...constructing translation matrices...')
    translate={}
    for i in range(len(fvs)):
      translate[tuple(fvs[i])]=sgs[i]
    standardPoints = {}
    for pt in fvs:
      stdPt = []
      for i,p in enumerate(pt):
        varName = self.sparseGrid.varNames[i]
        stdPt.append( self.distDict[varName].convertToQuad(self.quads[varName].type,p) )
      standardPoints[tuple(pt)] = stdPt[:]
    #make polynomials
    self.raiseADebug('...constructing polynomials...')
    self.norm = np.prod(list(self.distDict[v].measureNorm(self.quads[v].type) for v in self.distDict.keys()))
    for i,idx in enumerate(self.indexSet):
      idx=tuple(idx)
      for target in self.target:
        self.polyCoeffDict[target][idx]=0
        wtsum=0
        for pt,soln in zip(fvs,tvs[target]):
          tupPt = tuple(pt)
          stdPt = standardPoints[tupPt]
          wt = self.sparseGrid.weights(translate[tupPt])
          self.polyCoeffDict[target][idx]+=soln*self._multiDPolyBasisEval(idx,stdPt)*wt
        self.polyCoeffDict[target][idx]*=self.norm
    self.amITrained=True
    self.raiseADebug('...training complete!')

  def printPolyDict(self,printZeros=False):
    """
      Human-readable version of the polynomial chaos expansion.
      @ In, printZeros, bool, optional, optional flag for printing even zero coefficients
      @ Out, None
    """
    for target in self.target:
      data=[]
      for idx,val in self.polyCoeffDict[target].items():
        if abs(val) > 1e-12 or printZeros:
          data.append([idx,val])
      data.sort()
      self.raiseADebug('polyDict for ['+target+'] with inputs '+str(self.features)+':')
      for idx,val in data:
        self.raiseADebug('    '+str(idx)+' '+str(val))

  def checkForNonzeros(self,tol=1e-12):
    """
      Checks poly coefficient dictionary for nonzero entries.
      @ In, tol, float, optional, the tolerance under which is zero (default 1e-12)
      @ Out, data, dict, {'target1':list(tuple),'target2':list(tuple)}: the indices and values of the nonzero coefficients for each target
    """
    data = dict.fromkeys(self.target,[])
    for target in self.target:
      for idx,val in self.polyCoeffDict[target].items():
        if round(val,11) !=0:
          data[target].append([idx,val])
    return data

  def __mean__(self, targ=None):
    """
      Returns the mean of the ROM.
      @ In, None
      @ In, targ, str, optional, the target for which the __mean__ needs to be computed
      @ Out, __mean__, float, the mean
    """
    return self.__evaluateMoment__(1,targ)

  def __variance__(self, targ=None):
    """
      returns the variance of the ROM.
      @ In, None
      @ In, targ, str, optional, the target for which the __variance__ needs to be computed
      @ Out, __variance__, float, variance
    """
    mean = self.__evaluateMoment__(1,targ)
    return self.__evaluateMoment__(2,targ) - mean*mean

  def __evaluateMoment__(self,r, targ=None):
    """
      Use the ROM's built-in method to calculate moments.
      @ In, r, int, moment to calculate
      @ In, targ, str, optional, the target for which the moment needs to be computed
      @ Out, tot, float, evaluation of moment
    """
    target = self.target[0] if targ is None else targ
    #TODO is there a faster way still to do this?
    if r==1: return self.polyCoeffDict[target][tuple([0]*len(self.features))]
    elif r==2: return sum(s**2 for s in self.polyCoeffDict[target].values())
    tot=0
    for pt,wt in self.sparseGrid:
      tot+=self.__evaluateLocal__([pt])[target]**r*wt
    tot*=self.norm
    return tot

  def __evaluateLocal__(self,featureVals):
    """
      Evaluates a point.
      @ In, featureVals, list, of values at which to evaluate the ROM
      @ Out, returnDict, dict, the evaluated point for each target
    """
    featureVals=featureVals[0]
    returnDict={}
    stdPt = np.zeros(len(featureVals))
    for p,pt in enumerate(featureVals):
      varName = self.sparseGrid.varNames[p]
      stdPt[p] = self.distDict[varName].convertToQuad(self.quads[varName].type,pt)
    for target in self.target:
      tot=0
      for idx,coeff in self.polyCoeffDict[target].items():
        tot+=coeff*self._multiDPolyBasisEval(idx,stdPt)
      returnDict[target] = tot
    return returnDict

  def _printPolynomial(self):
    """
      Prints each polynomial for each coefficient.
      @ In, None
      @ Out, None
    """
    for target in self.target:
      self.raiseADebug('Target:'+target+'.Coeff Idx:')
      for idx,coeff in self.polyCoeffDict[target].items():
        if abs(coeff)<1e-12: continue
        self.raiseADebug(str(idx))
        for i,ix in enumerate(idx):
          var = self.features[i]
          self.raiseADebug(self.polys[var][ix]*coeff,'|',var)

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    params = {}
    return params

  def getSensitivities(self,targ=None):
    """
      Calculates the Sobol indices (percent partial variances) of the terms in this expansion.
      @ In, targ, str, optional, the target for which the moment needs to be computed
      @ Out, getSensitivities, tuple(dict), Sobol indices and partial variances keyed by subset
    """
    target = self.target[0] if targ is None else targ
    totVar = self.__variance__(target)
    partials = {}
    #calculate partial variances
    self.raiseADebug('Calculating partial variances...')
    for poly,coeff in self.polyCoeffDict[target].items():
      #use poly to determine subset
      subset = self._polyToSubset(poly)
      # skip mean
      if len(subset) < 1: continue
      subset = tuple(subset)
      if subset not in partials.keys():
        partials[subset] = 0
      partials[subset] += coeff*coeff
    #calculate Sobol indices
    indices = {}
    for subset,partial in partials.items():
      indices[subset] = partial / totVar
    return (indices,partials)

  def getTotalSensitivities(self,indices):
    """
      Given the Sobol global sensitivity indices, calculates the total indices for each subset.
      @ In, indices, dict, tuple(subset):float(index)
      @ Out, totals, dict, tuple(subset):float(index)
    """
    #total index is the sum of all Sobol indices in which a subset belongs
    totals={}
    for subset in indices.keys():
      setSub = set(subset)
      totals[subset] = 0
      for checkSubset in indices.keys():
        setCheck = set(checkSubset)
        if setSub.issubset(setCheck):
          totals[subset] += indices[checkSubset]
    return totals

  def _polyToSubset(self,poly):
    """
      Given a tuple with polynomial orders, returns the subset it belongs exclusively to
      @ In, poly, tuple(int), polynomial index set entry
      @ Out, subset, tuple(str), subset
    """
    boolRep = tuple(False if poly[i]==0 else True for i in range(len(poly)))
    subset = []
    for i,p in enumerate(boolRep):
      if p:
        subset.append(self.features[i])
    return tuple(subset)

#
#
#
#
class HDMRRom(GaussPolynomialRom):
  """
    High-Dimention Model Reduction reduced order model.  Constructs model based on subsets of the input space.
  """
  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals, 2-D numpy array, [n_samples,n_features]
      @ Out, confidence, float, the confidence
    """
    pass

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    pass

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    pass

  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    GaussPolynomialRom.__init__(self,messageHandler,**kwargs)
    self.initialized   = False #true only when self.initialize has been called
    self.printTag      = 'HDMR_ROM('+'-'.join(self.target)+')'
    self.sobolOrder    = None #depth of HDMR/Sobol expansion
    self.ROMs          = {}   #dict of GaussPolyROM objects keyed by combination of vars that make them up
    self.sdx           = None #dict of sobol sensitivity coeffs, keyed on order and tuple(varnames)
    self.mean          = None #mean, store to avoid recalculation
    self.variance      = None #variance, store to avoid recalculation
    self.anova         = None #converted true ANOVA terms, stores coefficients not polynomials
    self.partialVariances = None #partial variance contributions

    for key,val in kwargs.items():
      if key=='SobolOrder': self.sobolOrder = int(val)

  def _localPrintXML(self,outFile,pivotVal,options={}):
    """
      Adds requested entries to XML node.
      @ In, outFile, Files.File, either StaticXMLOutput or DynamicXMLOutput file
      @ In, pivotVal, float, value of pivot parameters to use in printing if dynamic
      @ In, options, dict, optional, dict of string-based options to use, including filename, things to print, etc
        May include:
        'what': comma-separated string list, the qualities to print out
        'pivotVal': float value of dynamic pivotParam value
      @ Out, None
    """
    #inherit from GaussPolynomialRom
    if not self.amITrained: self.raiseAnError(RuntimeError,'ROM is not yet trained!')
    self.mean=None
    canDo = ['mean','expectedValue','variance','samples','partialVariance','sobolIndices','sobolTotalIndices']
    if 'what' in options.keys():
      requests = list(o.strip() for o in options['what'].split(','))
      if 'all' in requests: requests = canDo
      #protect against things SCgPC can do that HDMR can't
      if 'polyCoeffs' in requests:
        self.raiseAWarning('HDMRRom cannot currently print polynomial coefficients.  Skipping...')
        requests.remove('polyCoeffs')
      options['what'] = ','.join(requests)
    else:
      self.raiseAWarning('No "what" options for XML printing are recognized!  Skipping...')
    GaussPolynomialRom._localPrintXML(self,outFile,pivotVal,options)

  def initialize(self,idict):
    """
      Initializes the instance.
      @ In, idict, dict, objects needed to initalize
      @ Out, None
    """
    for key,value in idict.items():
      if   key == 'ROMs'   : self.ROMs       = value
      elif key == 'dists'  : self.distDict   = value
      elif key == 'quads'  : self.quads      = value
      elif key == 'polys'  : self.polys      = value
      elif key == 'refs'   : self.references = value
      elif key == 'numRuns': self.numRuns    = value
    self.initialized = True

  def __trainLocal__(self,featureVals,targetVals):
    """
      Because HDMR rom is a collection of sub-roms, we call sub-rom "train" to do what we need it do.
      @ In, featureVals, np.array, training feature values
      @ In, targetVals, np.array, training target values
      @ Out, None
    """
    if not self.initialized:
      self.raiseAnError(RuntimeError,'ROM has not yet been initialized!  Has the Sampler associated with this ROM been used?')
    ft={}
    self.refSoln = {key:dict({}) for key in self.target}
    for i in range(len(featureVals)):
      ft[tuple(featureVals[i])]=targetVals[i,:]

    #get the reference case
    self.refpt = tuple(self.__fillPointWithRef((),[]))
    for cnt, target in enumerate(self.target):
      self.refSoln[target] = ft[self.refpt][cnt]
    for combo,rom in self.ROMs.items():
      subtdict = {key:list([]) for key in self.target}
      for c in combo: subtdict[c]=[]
      SG = rom.sparseGrid
      fvals=np.zeros([len(SG),len(combo)])
      tvals=np.zeros((len(SG),len(self.target)))
      for i in range(len(SG)):
        getpt=tuple(self.__fillPointWithRef(combo,SG[i][0]))
        #the 1e-10 is to be consistent with RAVEN's CSV print precision
        tvals[i,:] = ft[tuple(mathUtils.NDInArray(np.array(ft.keys()),getpt,tol=1e-10)[2])]
        for fp,fpt in enumerate(SG[i][0]):
          fvals[i][fp] = fpt
      for i,c in enumerate(combo):
        subtdict[c] = fvals[:,i]
      for cnt, target in enumerate(self.target):
        subtdict[target] = tvals[:,cnt]
      rom.train(subtdict)

    #make ordered list of combos for use later
    maxLevel = max(list(len(combo) for combo in self.ROMs.keys()))
    self.combos = []
    for i in range(maxLevel+1):
      self.combos.append([])
    for combo in self.ROMs.keys():
      self.combos[len(combo)].append(combo)

    #list of term objects
    self.terms = {():[]}  # each entry will look like 'x1,x2':('x1','x2'), missing the reference entry
    for l in range(1,maxLevel+1):
      for romName in self.combos[l]:
        self.terms[romName] = []
        # add subroms -> does this get referenece case, too?
        for key in self.terms.keys():
          if set(key).issubset(set(romName)) and key!=romName:
            self.terms[romName].append(key)
    #reduce terms
    self.reducedTerms = {}
    for term in self.terms.keys():
      self._collectTerms(term,self.reducedTerms)
    #remove zero entries
    self._removeZeroTerms(self.reducedTerms)

    self.amITrained = True

  def __fillPointWithRef(self,combo,pt):
    """
      Given a "combo" subset of the full input space and a partially-filled
      point within that space, fills the rest of space with the reference
      cut values.
      @ In, combo, tuple(str), names of subset dimensions
      @ In, pt, list(float), values of points in subset dimension
      @ Out, newpt, tuple(float), full point in input dimension space on cut-hypervolume
    """
    newpt=np.zeros(len(self.features))
    for v,var in enumerate(self.features):
      if var in combo:
        newpt[v] = pt[combo.index(var)]
      else:
        newpt[v] = self.references[var]
    return newpt

  def __fillIndexWithRef(self,combo,pt):
    """
       Given a "combo" subset of the full input space and a partially-filled
       polynomial order index within that space, fills the rest of index with zeros.
       @ In, combo, tuple of strings, names of subset dimensions
       @ In, pt, list of floats, values of points in subset dimension
       @ Out, newpt, tuple(int), full index in input dimension space on cut-hypervolume
    """
    newpt=np.zeros(len(self.features),dtype=int)
    for v,var in enumerate(self.features):
      if var in combo:
        newpt[v] = pt[combo.index(var)]
    return tuple(newpt)

  def __evaluateLocal__(self,featureVals):
    """
      Evaluates a point.
      @ In, featureVals, list(float), list of values at which to evaluate the ROM
      @ Out, tot, float, the evaluated point
    """
    #am I trained?
    returnDict = dict.fromkeys(self.target,None)
    if not self.amITrained: self.raiseAnError(IOError,'Cannot evaluate, as ROM is not trained!')
    for target in self.target:
      tot = 0
      for term,mult in self.reducedTerms.items():
        if term == ():
          tot += self.refSoln[target]*mult
        else:
          cutVals = [list(featureVals[0][self.features.index(j)] for j in term)]
          tot += self.ROMs[term].__evaluateLocal__(cutVals)[target]*mult
      returnDict[target] = tot
    return returnDict

  def __mean__(self,targ=None):
    """
      The Cut-HDMR approximation can return its mean easily.
      @ In, targ, str, optional, the target for which the __mean__ needs to be computed
      @ Out, __mean__, float, the mean
    """
    if not self.amITrained: self.raiseAnError(IOError,'Cannot evaluate mean, as ROM is not trained!')
    return self._calcMean(self.reducedTerms,targ)

  def __variance__(self,targ=None):
    """
      The Cut-HDMR approximation can return its variance somewhat easily.
      @ In, targ, str, optional, the target for which the __mean__ needs to be computed
      @ Out, __variance__, float, the variance
    """
    if not self.amITrained: self.raiseAnError(IOError,'Cannot evaluate variance, as ROM is not trained!')
    target = self.target[0] if targ is None else targ
    self.getSensitivities(target)
    return sum(val for val in self.partialVariances[target].values())

  def _calcMean(self,fromDict,targ=None):
    """
      Given a subset, calculate mean from terms
      @ In, fromDict, dict{string:int}, ROM subsets and their multiplicity
      @ In, targ, str, optional, the target for which the __mean__ needs to be computed
      @ Out, tot, float, mean
    """
    tot = 0
    for term,mult in fromDict.items():
      tot += self._evaluateIntegral(term,targ)*mult
    return tot

  def _collectTerms(self,a,targetDict,sign=1,depth=0):
    """
      Adds main term multiplicity and subtracts sub term multiplicity for cross between terms
      @ In, targetDict, dict, dictionary to pace terms in
      @ In, a, string, main combo key from self.terms
      @ In, sign, int, optional, gives the signs of the terms (1 for positive, -1 for negative)
      @ In, depth, int, optional, recursion depth
      @ Out, None
    """
    if a not in targetDict.keys(): targetDict[a] = sign
    else: targetDict[a] += sign
    for sub in self.terms[a]:
      self._collectTerms(sub,targetDict,sign*-1,depth+1)

  def _evaluateIntegral(self,term, targ=None):
    """
      Uses properties of orthonormal gPC to algebraically evaluate integrals gPC
      This does assume the integral is over all the constituent variables in the the term
      @ In, term, string, subset term to integrate
      @ In, targ, str, optional, the target for which the __mean__ needs to be computed
      @ Out, _evaluateIntegral, float, evaluation

    """
    if term in [(),'',None]:
      return self.refSoln[targ if targ is not None else self.target[0]]
    else:
      return self.ROMs[term].__evaluateMoment__(1,targ)

  def _removeZeroTerms(self,d):
    """
      Removes keys from d that have zero value
      @ In, d, dict, string:int
      @ Out, None
    """
    toRemove=[]
    for key,val in d.items():
      if abs(val) < 1e-15: toRemove.append(key)
    for rem in toRemove:
      del d[rem]

  def getSensitivities(self,targ=None):
    """
      Calculates the Sobol indices (percent partial variances) of the terms in this expansion.
      @ In, targ, str, optional, the target for which the moment needs to be computed
      @ Out, getSensitivities, tuple(dict), Sobol indices and partial variances keyed by subset
    """
    target = self.target[0] if targ is None else targ
    if self.sdx is not None and self.partialVariances is not None and target in self.sdx.keys():
      self.raiseADebug('Using previously-constructed ANOVA terms...')
      return self.sdx[target],self.partialVariances[target]
    self.raiseADebug('Constructing ANOVA terms...')
    #collect terms
    terms = {}
    allFalse = tuple(False for _ in self.features)
    for subset,mult in self.reducedTerms.items():
      #skip mean, since it will be subtracted off in the end
      if subset == (): continue
      for poly,coeff in self.ROMs[subset].polyCoeffDict[target].items():
        #skip mean terms
        if sum(poly) == 0: continue
        poly = self.__fillIndexWithRef(subset,poly)
        polySubset = self._polyToSubset(poly)
        if polySubset not in terms.keys(): terms[polySubset] = {}
        if poly not in terms[polySubset].keys(): terms[polySubset][poly] = 0
        terms[polySubset][poly] += coeff*mult
    #calculate partial variances
    self.partialVariances = {target: dict({})}
    self.sdx              = {target: dict({})}
    for subset in terms.keys():
      self.partialVariances[target][subset] = sum(v*v for v in terms[subset].values())
    #calculate indices
    totVar = sum(self.partialVariances[target].values())
    for subset,value in self.partialVariances[target].items():
      self.sdx[target][subset] = value / totVar
    return self.sdx[target],self.partialVariances[target]

#
#
#
#
class pickledROM(superVisedLearning):
  """
    Placeholder for ROMs that will be generated by unpickling from file.
  """
  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    self.printTag          = 'pickledROM'
    self.messageHandler    = messageHandler
    self._dynamicHandling  = False
    self.initOptionDict    = {}
    self.features          = ['PlaceHolder']
    self.target            = 'PlaceHolder'

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals, 2-D numpy array, [n_samples,n_features]
      @ Out, confidence, float, the confidence
    """
    pass

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    pass

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    pass

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    params = {}
    return params

  def __evaluateLocal__(self,featureVals):
    """
      Evaluates a point.
      @ In, featureVals, list, of values at which to evaluate the ROM
      @ Out, returnDict, dict, the evaluated point for each target
    """
    self.raiseAnError(RuntimeError, 'PickledROM has not been loaded from file yet!  An IO step is required to perform this action.')

  def __trainLocal__(self,featureVals,targetVals):
    """
      Trains ROM.
      @ In, featureVals, np.ndarray, feature values
      @ In, targetVals, np.ndarray, target values
    """
    self.raiseAnError(RuntimeError, 'PickledROM has not been loaded from file yet!  An IO step is required to perform this action.')
#
#
#
#
class MSR(NDinterpolatorRom):
  """
    MSR class - Computes an approximated hierarchical Morse-Smale decomposition
    from an input point cloud consisting of an arbitrary number of input
    parameters and one or more response values per input point
  """
  def __init__(self, messageHandler, **kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    self.printTag = 'MSR ROM'
    superVisedLearning.__init__(self,messageHandler,**kwargs)
    self.acceptedGraphParam = ['approximate knn', 'delaunay', 'beta skeleton', \
                               'relaxed beta skeleton']
    self.acceptedPersistenceParam = ['difference','probability','count','area']
    self.acceptedGradientParam = ['steepest', 'maxflow']
    self.acceptedNormalizationParam = ['feature', 'zscore', 'none']
    self.acceptedPredictorParam = ['kde', 'svm']
    self.acceptedKernelParam = ['uniform', 'triangular', 'epanechnikov',
                                'biweight', 'quartic', 'triweight', 'tricube',
                                'gaussian', 'cosine', 'logistic', 'silverman',
                                'exponential']
    self.__amsc = []                      # AMSC object
    # Some sensible default arguments
    self.gradient = 'steepest'            # Gradient estimate methodology
    self.graph = 'beta skeleton'          # Neighborhood graph used
    self.beta = 1                         # beta used in the beta skeleton graph
                                          #  and its relaxed version
    self.knn = -1                         # k-nearest neighbor value for either
                                          #  the approximate knn strategy, or
                                          #  for initially pruning the beta
                                          #  skeleton graphs. (this could also
                                          #  potentially be used for restricting
                                          #  the models influencing a query
                                          #  point to only use those models
                                          #  belonging to a limited
                                          #  neighborhood of training points)
    self.simplification = 0               # Morse-smale simplification amount
                                          #  this should probably be normalized
                                          #  to [0,1], however for now it is not
                                          #  and the scale of it will depend on
                                          #  the type of persistence used
    self.persistence = 'difference'       # Strategy for merging topo partitions
    self.weighted = False                 # Should the linear models be weighted
                                          #  by probability information?
    self.normalization = None             # Should any normalization be
                                          #  performed within the AMSC? No, this
                                          #  data should already be standardized
    self.partitionPredictor = 'kde'       # The method used to predict the label
                                          #  of each query point (can be soft).
    self.blending = False                 # Flag: blend the predictions
                                          #  depending on soft label predictions
                                          #  or use only the most likely local
                                          #  model
    self.kernel = 'gaussian'              # What kernel should be used in the
                                          #  kde approach
    self.bandwidth = 1.                   # The bandwidth for the kde approach

    # Read everything in first, and then do error checking as some parameters
    # will not matter, but we can still throw a warning message that they may
    # want to clean up there input file. In some cases, we will have to do
    # value checking in place since the type cast can fail.
    for key,val in kwargs.items():
      if key.lower() == 'graph':
        self.graph = val.strip().encode('ascii').lower()
      elif key.lower() == "gradient":
        self.gradient = val.strip().encode('ascii').lower()
      elif key.lower() == "beta":
        try:
          self.beta = float(val)
        except ValueError:
          # If the user has specified a graph, use it, otherwise be sure to use
          #  the default when checking whether this is a warning or an error
          if 'graph' in kwargs:
            graph = kwargs['graph'].strip().encode('ascii').lower()
          else:
            graph = self.graph
          if graph.endswith('beta skeleton'):
            self.raiseAnError(IOError, 'Requested invalid beta value:',
                              val, '(Allowable range: (0,2])')
          else:
            self.raiseAWarning('Requested invalid beta value:', self.beta,
                               '(Allowable range: (0,2]), however beta is',
                               'ignored when using the', graph,
                               'graph structure.')
      elif key.lower() == 'knn':
        try:
          self.knn = int(val)
        except ValueError:
          self.raiseAnError(IOError, 'Requested invalid knn value:',
                            val, '(Should be an integer value, knn <= 0 implies'
                            ,'use of the fully connected point set)')
      elif key.lower() == 'simplification':
        try:
          self.simplification = float(val)
        except ValueError:
          self.raiseAnError(IOError, 'Requested invalid simplification level:',
                            val, '(should be a floating point value)')
      elif key.lower() == 'bandwidth':
        if val == 'variable' or val == 'auto':
          self.bandwidth = val
        else:
          try:
            self.bandwidth = float(val)
          except ValueError:
            # If the user has specified a strategy, use it, otherwise be sure to
            #  use the default when checking whether this is a warning or an error
            if 'partitionPredictor' in kwargs:
              partPredictor = kwargs['partitionPredictor'].strip().encode('ascii').lower()
            else:
              partPredictor = self.partitionPredictor
            if partPredictor == 'kde':
              self.raiseAnError(IOError, 'Requested invalid bandwidth value:',
                                val,'(should be a positive floating point value)')
            else:
              self.raiseAWarning('Requested invalid bandwidth value:',val,
                                 '(bandwidth > 0 or \"variable\"). However, it is ignored when',
                                 'using the', partPredictor, 'partition',
                                 'predictor')
      elif key.lower() == 'persistence':
        self.persistence = val.strip().encode('ascii').lower()
      elif key.lower() == 'partitionpredictor':
        self.partitionPredictor = val.strip().encode('ascii').lower()
      elif key.lower() == 'smooth':
        self.blending = True
      elif key.lower() == "kernel":
        self.kernel = val
      else:
        pass

    # Morse-Smale specific error handling
    if self.graph not in self.acceptedGraphParam:
      self.raiseAnError(IOError, 'Requested unknown graph type:',
                        '\"'+self.graph+'\"','(Available options:',
                        self.acceptedGraphParam,')')
    if self.gradient not in self.acceptedGradientParam:
      self.raiseAnError(IOError, 'Requested unknown gradient method:',
                        '\"'+self.gradient+'\"', '(Available options:',
                        self.acceptedGradientParam,')')
    if self.beta <= 0 or self.beta > 2:
      if self.graph.endswith('beta skeleton'):
        self.raiseAnError(IOError, 'Requested invalid beta value:',
                          self.beta, '(Allowable range: (0,2])')
      else:
        self.raiseAWarning('Requested invalid beta value:', self.beta,
                           '(Allowable range: (0,2]), however beta is',
                           'ignored when using the', self.graph,
                           'graph structure.')
    if self.persistence not in self.acceptedPersistenceParam:
      self.raiseAnError(IOError, 'Requested unknown persistence method:',
                        '\"'+self.persistence+'\"', '(Available options:',
                        self.acceptedPersistenceParam,')')
    if self.partitionPredictor not in self.acceptedPredictorParam:
      self.raiseAnError(IOError, 'Requested unknown partition predictor:'
                        '\"'+self.partitionPredictor+'\"','(Available options:',
                        self.acceptedPredictorParam,')')
    if self.bandwidth <= 0:
      if self.partitionPredictor == 'kde':
        self.raiseAnError(IOError, 'Requested invalid bandwidth value:',
                          self.bandwidth, '(bandwidth > 0)')
      else:
        self.raiseAWarning(IOError, 'Requested invalid bandwidth value:',
                          self.bandwidth, '(bandwidth > 0). However, it is',
                          'ignored when using the', self.partitionPredictor,
                          'partition predictor')

    if self.kernel not in self.acceptedKernelParam:
      if self.partitionPredictor == 'kde':
        self.raiseAnError(IOError, 'Requested unknown kernel:',
                          '\"'+self.kernel+'\"', '(Available options:',
                          self.acceptedKernelParam,')')
      else:
        self.raiseAWarning('Requested unknown kernel:', '\"'+self.kernel+'\"',
                           '(Available options:', self.acceptedKernelParam,
                           '), however the kernel is ignored when using the',
                           self.partitionPredictor,'partition predictor.')
    self.__resetLocal__()

  def __getstate__(self):
    """
      Overwrite state (for pickle-ing)
      we do not pickle the HDF5 (C++) instance
      but only the info to re-load it
      @ In, None
      @ Out, state, dict, namespace dictionary
    """
    state = dict(self.__dict__)
    state.pop('_MSR__amsc')
    state.pop('kdTree')
    return state

  def __setstate__(self,newState):
    """
      Initialize the ROM with the data contained in newstate
      @ In, newState, dict, it contains all the information needed by the ROM to be initialized
      @ Out, None
    """
    for key, value in newState.iteritems():
      setattr(self, key, value)
    self.kdTree             = None
    self.__amsc             = []
    self.__trainLocal__(self.X,self.Y)

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on samples in featureVals with responses y.
      @ In, featureVals, np.ndarray or list of list, shape=[n_samples, n_features],
        an array of input feature values
      @ In, targetVals, np.ndarray, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """

    # # Possibly load this here in case people have trouble building it, so it
    # # only errors if they try to use it?
    from AMSC_Object import AMSC_Object

    self.X = featureVals[:][:]
    self.Y = targetVals

    if self.weighted:
      self.raiseAnError(NotImplementedError,
                    ' cannot use weighted data right now.')
    else:
      weights = None

    if self.knn <= 0:
      self.knn = self.X.shape[0]

    names = [name.encode('ascii') for name in self.features + self.target]
    # Data is already normalized, so ignore this parameter
    ### Comment replicated from the post-processor version, not sure what it
    ### means (DM)
    # FIXME: AMSC_Object employs unsupervised NearestNeighbors algorithm from
    #        scikit learn.
    #        The NearestNeighbor algorithm is implemented in
    #        SupervisedLearning, which requires features and targets by
    #        default, which we don't have here. When the NearestNeighbor is
    #        implemented in unSupervisedLearning switch to it.
    for index in range(len(self.target)):
      self.__amsc.append( AMSC_Object(X=self.X, Y=self.Y[:,index], w=weights, names=names,
                                      graph=self.graph, gradient=self.gradient,
                                      knn=self.knn, beta=self.beta,
                                      normalization=None,
                                      persistence=self.persistence) )
      self.__amsc[index].Persistence(self.simplification)
      self.__amsc[index].BuildLinearModels(self.simplification)

    # We need a KD-Tree for querying neighbors
    self.kdTree = neighbors.KDTree(self.X)

    distances,_ = self.kdTree.query(self.X,k=self.knn)
    distances = distances.flatten()

    # The following are a list of common kernels defined centered at zero with
    # either infinite support or a support defined over the interval [1,1].
    # See: https://en.wikipedia.org/wiki/Kernel_(statistics)
    # Thus, the use of this indicator function. When using these kernels, we
    # must be sure to first scale the parameter into this support before calling
    # it. In our case, we want to center our information, such that the maximum
    # value occurs when the two points coincide, and so we will set u to be
    # inversely proportional to the distance between two points, and scaled by
    # a bandwidth parameter (either the user will fix, or we will compute)
    def indicator(u):
      """
        Method to return the indicator (see explaination above)
        @ In, u, float, the value to inquire
        @ Out, indicator, float, the abs of u
      """
      return np.abs(u)<1

    if self.kernel == 'uniform':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Uniform kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return 0.5*indicator(u)
    elif self.kernel == 'triangular':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Triangular kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return (1-abs(u))*indicator(u)
    elif self.kernel == 'epanechnikov':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Epanechnikov kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return ( 3./4. )*(1-u**2)*indicator(u)
    elif self.kernel == 'biweight' or self.kernel == 'quartic':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Biweight kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return (15./16.)*(1-u**2)**2*indicator(u)
    elif self.kernel == 'triweight':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Triweight kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return (35./32.)*(1-u**2)**3*indicator(u)
    elif self.kernel == 'tricube':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Tricube kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return (70./81.)*(1-abs(u)**3)**3*indicator(u)
    elif self.kernel == 'gaussian':
      if self.bandwidth == 'auto':
        self.bandwidth = 1.06*distances.std()*len(distances)**(-1./5.)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Gaussian kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return 1./np.sqrt(2*math.pi)*np.exp(-0.5*u**2)
    elif self.kernel == 'cosine':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Cosine kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return math.pi/4.*math.cos(u*math.pi/2.)*indicator(u)
    elif self.kernel == 'logistic':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Logistic kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return 1./(np.exp(u)+2+np.exp(-u))
    elif self.kernel == 'silverman':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Silverman kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        sqrt2 = math.sqrt(2)
        return 0.5 * np.exp(-abs(u)/sqrt2) * np.sin(abs(u)/sqrt2+math.pi/4.)
    elif self.kernel == 'exponential':
      if self.bandwidth == 'auto':
        self.bandwidth = max(distances)
        self.raiseAWarning('automatic bandwidth not yet implemented for the'
                           + self.kernel + ' kernel.')
      def kernel(u):
        """
          Exponential kernel
          @ In, u, float, the support
          @ Out, kernel, float, the kernel
        """
        return np.exp(-abs(u))
    self.__kernel = kernel

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      Should return distance to nearest neighbor or average prediction error of
      all neighbors?
      @ In, featureVals, 2-D numpy array [n_samples,n_features]
      @ Out, confidence, float, the confidence
    """
    self.raiseAnError(NotImplementedError, '__confidenceLocal__ method must be implemented!')

  def __evaluateLocal__(self,featureVals):
    """
      Perform regression on samples in featureVals.
      This will use the local predictor of each neighboring point weighted by its
      distance to that point.
      @ In, featureVals, numpy.array 2-D, features
      @ Out, returnDict, dict, dict of predicted values for each target ({'target1':numpy.array 1-D,'target2':numpy.array 1-D}
    """
    returnDict = {}
    for index, target in enumerate(self.target):
      if self.partitionPredictor == 'kde':
        partitions = self.__amsc[index].Partitions(self.simplification)
        weights = {}
        dists = np.zeros((featureVals.shape[0],self.X.shape[0]))
        for i,row in enumerate(featureVals):
          dists[i] = np.sqrt(((row-self.X)**2).sum(axis=-1))
        # This is a variable-based bandwidth that will adjust to the density
        # around the given query point
        if self.bandwidth == 'variable':
          h = sorted(dists)[self.knn-1]
        else:
          h = self.bandwidth
        for key,indices in partitions.iteritems():
          #############
          ## Using SciKit Learn, we have a limited number of kernel functions to
          ## choose from.
          # kernel = self.kernel
          # if kernel == 'uniform':
          #   kernel = 'tophat'
          # if kernel == 'triangular':
          #   kernel = 'linear'
          # kde = KernelDensity(kernel=kernel, bandwidth=h).fit(self.X[indices,])
          # weights[key] = np.exp(kde.score_samples(featureVals))
          #############
          ## OR
          #############
          weights[key] = 0
          for idx in indices:
            weights[key] += self.__kernel(dists[:,idx]/h)
          weights[key]
          #############

        if self.blending:
          weightedPredictions = np.zeros(featureVals.shape[0])
          sumW = 0
          for key in partitions.keys():
            fx = self.__amsc[index].Predict(featureVals,key)
            wx = weights[key]
            sumW += wx
            weightedPredictions += fx*wx
          returnDict[target] = weightedPredictions if sumW == 0 else weightedPredictions / sumW
        else:
          predictions = np.zeros(featureVals.shape[0])
          maxWeights = np.zeros(featureVals.shape[0])
          for key in partitions.keys():
            fx = self.__amsc[index].Predict(featureVals,key)
            wx = weights[key]
            predictions[wx > maxWeights] = fx
            maxWeights[wx > maxWeights] = wx
          returnDict[target] = predictions
      elif self.partitionPredictor == 'svm':
        partitions = self.__amsc[index].Partitions(self.simplification)
        labels = np.zeros(self.X.shape[0])
        for idx,(key,indices) in enumerate(partitions.iteritems()):
          labels[np.array(indices)] = idx
        # In order to make this deterministic for testing purposes, let's fix
        # the random state of the SVM object. Maybe, this could be exposed to the
        # user, but it shouldn't matter too much what the seed is for this.
        svc = svm.SVC(probability=True,random_state=np.random.RandomState(8),tol=1e-15)
        svc.fit(self.X,labels)
        probabilities = svc.predict_proba(featureVals)

        classIdxs = list(svc.classes_)
        if self.blending:
          weightedPredictions = np.zeros(len(featureVals))
          sumW = 0
          for idx,key in enumerate(partitions.keys()):
            fx = self.__amsc[index].Predict(featureVals,key)
            # It could be that a particular partition consists of only the extrema
            # and they themselves point to cells with different opposing extrema.
            # That is, a maximum points to a different minimum than the minimum in
            # the two point partition. Long story short, we need to be prepared for
            # an empty partition which will thus not show up in the predictions of
            # the SVC, since no point has it as a label.
            if idx not in classIdxs:
              wx = np.zeros(probabilities.shape[0])
            else:
              realIdx = list(svc.classes_).index(idx)
              wx = probabilities[:,realIdx]
            if self.blending:
              weightedPredictions = weightedPredictions + fx*wx
              sumW += wx
          returnDict[target] = weightedPredictions if sumW == 0 else weightedPredictions / sumW
        else:
          predictions = np.zeros(featureVals.shape[0])
          maxWeights = np.zeros(featureVals.shape[0])
          for idx,key in enumerate(partitions.keys()):
            fx = self.__amsc[index].Predict(featureVals,key)
            # It could be that a particular partition consists of only the extrema
            # and they themselves point to cells with different opposing extrema.
            # That is, a maximum points to a different minimum than the minimum in
            # the two point partition. Long story short, we need to be prepared for
            # an empty partition which will thus not show up in the predictions of
            # the SVC, since no point has it as a label.
            if idx not in classIdxs:
              wx = np.zeros(probabilities.shape[0])
            else:
              realIdx = list(svc.classes_).index(idx)
              wx = probabilities[:,realIdx]
            predictions[wx > maxWeights] = fx
            maxWeights[wx > maxWeights] = wx
          returnDict[target] = predictions
      return returnDict


  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    self.X      = []
    self.Y      = []
    self.__amsc = []
    self.kdTree = None


#
#
#
class NDsplineRom(NDinterpolatorRom):
  """
    An N-dimensional Spline model
  """
  ROMtype         = 'NDsplineRom'
  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    NDinterpolatorRom.__init__(self,messageHandler,**kwargs)
    self.printTag = 'ND-SPLINE ROM'
    for _ in range(len(self.target)):
      self.interpolator.append(interpolationND.NDspline())

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    for index in range(len(self.target)):
      self.interpolator[index].reset()
#
#
#
class NDinvDistWeight(NDinterpolatorRom):
  """
    An N-dimensional model that interpolates data based on a inverse weighting of
    their training data points?
  """
  ROMtype         = 'NDinvDistWeight'
  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    NDinterpolatorRom.__init__(self,messageHandler,**kwargs)
    self.printTag = 'ND-INVERSEWEIGHT ROM'
    if not 'p' in self.initOptionDict.keys(): self.raiseAnError(IOError,'the <p> parameter must be provided in order to use NDinvDistWeigth as ROM!!!!')
    self.__initLocal__()

  def __initLocal__(self):
    """
      Method used to add additional initialization features used by pickling
      @ In, None
      @ Out, None
    """
    self.interpolator = []
    for _ in range(len(self.target)):
      self.interpolator.append(interpolationND.InverseDistanceWeighting(float(self.initOptionDict['p'])))

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    for index in range(len(self.target)):
      self.interpolator[index].reset(float(self.initOptionDict['p']))
#
#
#
class SciKitLearn(superVisedLearning):
  """
    An Interface to the ROMs provided by skLearn
  """
  # the normalization strategy is defined through the Boolean value in the dictionary below:
  # {mainClass:{subtype:(classPointer,Output type (float or int), boolean -> External Z-normalization needed)}
  ROMtype = 'SciKitLearn'

  ## This seems more manual than it needs to be, why not use something like:
  # import os, sys, pkgutil, inspect
  # import sklearn

  # # sys.modules['sklearn']
  # sklearnSubmodules = [name for _,name,_ in pkgutil.iter_modules([os.path.dirname(sklearn.__file__)])]
  # loadedSKL = []

  # sklLibrary = {}
  # for key,mod in globals().items():
  #   if key in sklearnSubmodules:
  #     members = inspect.getmembers(mod, inspect.isclass)
  #     for mkey,member in members:
  #       sklLibrary[key+'|'+mkey] = member

  ## Now sklLibrary holds keys that are the same as what the user inputs, and
  ## the values are the classes that can be directly instantiated. One would
  ## still need the float/int and boolean designations, but my suspicion is that
  ## these "special" cases are just not being handled in a generic enough
  ## fashion. This doesn't seem like something we should be tracking.


  availImpl                                                 = {}                                                            # dictionary of available ROMs {mainClass:{subtype:(classPointer,Output type (float or int), boolean -> External Z-normalization needed)}
  availImpl['lda']                                          = {}                                                            #Linear Discriminant Analysis
  availImpl['lda']['LDA']                                   = (lda.LDA                                  , 'int'    , False) #Quadratic Discriminant Analysis (QDA)
  # availImpl['lda']['LDA']                                   = (discriminant_analysis.LinearDiscriminantAnalysis, 'int'    , False) #Quadratic Discriminant Analysis (QDA)
  availImpl['linear_model']                                 = {}                                                            #Generalized Linear Models
  availImpl['linear_model']['ARDRegression'               ] = (linear_model.ARDRegression               , 'float'  , False) #Bayesian ARD regression.
  availImpl['linear_model']['BayesianRidge'               ] = (linear_model.BayesianRidge               , 'float'  , False) #Bayesian ridge regression
  availImpl['linear_model']['ElasticNet'                  ] = (linear_model.ElasticNet                  , 'float'  , False) #Linear Model trained with L1 and L2 prior as regularizer
  availImpl['linear_model']['ElasticNetCV'                ] = (linear_model.ElasticNetCV                , 'float'  , False) #Elastic Net model with iterative fitting along a regularization path
  availImpl['linear_model']['Lars'                        ] = (linear_model.Lars                        , 'float'  , False) #Least Angle Regression model a.k.a.
  availImpl['linear_model']['LarsCV'                      ] = (linear_model.LarsCV                      , 'float'  , False) #Cross-validated Least Angle Regression model
  availImpl['linear_model']['Lasso'                       ] = (linear_model.Lasso                       , 'float'  , False) #Linear Model trained with L1 prior as regularizer (aka the Lasso)
  availImpl['linear_model']['LassoCV'                     ] = (linear_model.LassoCV                     , 'float'  , False) #Lasso linear model with iterative fitting along a regularization path
  availImpl['linear_model']['LassoLars'                   ] = (linear_model.LassoLars                   , 'float'  , False) #Lasso model fit with Least Angle Regression a.k.a.
  availImpl['linear_model']['LassoLarsCV'                 ] = (linear_model.LassoLarsCV                 , 'float'  , False) #Cross-validated Lasso, using the LARS algorithm
  availImpl['linear_model']['LassoLarsIC'                 ] = (linear_model.LassoLarsIC                 , 'float'  , False) #Lasso model fit with Lars using BIC or AIC for model selection
  availImpl['linear_model']['LinearRegression'            ] = (linear_model.LinearRegression            , 'float'  , False) #Ordinary least squares Linear Regression.
  availImpl['linear_model']['LogisticRegression'          ] = (linear_model.LogisticRegression          , 'float'  , True ) #Logistic Regression (aka logit, MaxEnt) classifier.
  availImpl['linear_model']['MultiTaskLasso'              ] = (linear_model.MultiTaskLasso              , 'float'  , False) #Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer
  availImpl['linear_model']['MultiTaskElasticNet'         ] = (linear_model.MultiTaskElasticNet         , 'float'  , False) #Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer
  availImpl['linear_model']['OrthogonalMatchingPursuit'   ] = (linear_model.OrthogonalMatchingPursuit   , 'float'  , False) #Orthogonal Mathching Pursuit model (OMP)
  availImpl['linear_model']['OrthogonalMatchingPursuitCV' ] = (linear_model.OrthogonalMatchingPursuitCV , 'float'  , False) #Cross-validated Orthogonal Mathching Pursuit model (OMP)
  availImpl['linear_model']['PassiveAggressiveClassifier' ] = (linear_model.PassiveAggressiveClassifier , 'int'    , True ) #Passive Aggressive Classifier
  availImpl['linear_model']['PassiveAggressiveRegressor'  ] = (linear_model.PassiveAggressiveRegressor  , 'float'  , True ) #Passive Aggressive Regressor
  availImpl['linear_model']['Perceptron'                  ] = (linear_model.Perceptron                  , 'float'  , True ) #Perceptron
  availImpl['linear_model']['Ridge'                       ] = (linear_model.Ridge                       , 'float'  , False) #Linear least squares with l2 regularization.
  availImpl['linear_model']['RidgeClassifier'             ] = (linear_model.RidgeClassifier             , 'float'  , False) #Classifier using Ridge regression.
  availImpl['linear_model']['RidgeClassifierCV'           ] = (linear_model.RidgeClassifierCV           , 'int'    , False) #Ridge classifier with built-in cross-validation.
  availImpl['linear_model']['RidgeCV'                     ] = (linear_model.RidgeCV                     , 'float'  , False) #Ridge regression with built-in cross-validation.
  availImpl['linear_model']['SGDClassifier'               ] = (linear_model.SGDClassifier               , 'int'    , True ) #Linear classifiers (SVM, logistic regression, a.o.) with SGD training.
  availImpl['linear_model']['SGDRegressor'                ] = (linear_model.SGDRegressor                , 'float'  , True ) #Linear model fitted by minimizing a regularized empirical loss with SGD

  availImpl['svm']                                          = {}                                                            #support Vector Machines
  availImpl['svm']['LinearSVC'                            ] = (svm.LinearSVC                            , 'bool'   , True ) #Linear Support vector classifier
  availImpl['svm']['SVC'                                  ] = (svm.SVC                                  , 'bool'   , True ) #Support vector classifier
  availImpl['svm']['NuSVC'                                ] = (svm.NuSVC                                , 'bool'   , True ) #Nu Support vector classifier
  availImpl['svm']['SVR'                                  ] = (svm.SVR                                  , 'float'  , True ) #Support vector regressor

  availImpl['multiClass']                                   = {} #Multiclass and multilabel classification
  availImpl['multiClass']['OneVsRestClassifier'           ] = (multiclass.OneVsRestClassifier           , 'int'   ,  False) # One-vs-the-rest (OvR) multiclass/multilabel strategy
  availImpl['multiClass']['OneVsOneClassifier'            ] = (multiclass.OneVsOneClassifier            , 'int'   ,  False) # One-vs-one multiclass strategy
  availImpl['multiClass']['OutputCodeClassifier'          ] = (multiclass.OutputCodeClassifier          , 'int'   ,  False) # (Error-Correcting) Output-Code multiclass strategy

  availImpl['naiveBayes']                                   = {}
  availImpl['naiveBayes']['GaussianNB'                    ] = (naive_bayes.GaussianNB                   , 'float' ,  True )
  availImpl['naiveBayes']['MultinomialNB'                 ] = (naive_bayes.MultinomialNB                , 'float' ,  False)
  availImpl['naiveBayes']['BernoulliNB'                   ] = (naive_bayes.BernoulliNB                  , 'float' ,  True )

  availImpl['neighbors']                                    = {}
  availImpl['neighbors']['KNeighborsClassifier'           ] = (neighbors.KNeighborsClassifier           , 'int'   ,  True )# Classifier implementing the k-nearest neighbors vote.
  availImpl['neighbors']['RadiusNeighbors'                ] = (neighbors.RadiusNeighborsClassifier      , 'int'   ,  True )# Classifier implementing a vote among neighbors within a given radius
  availImpl['neighbors']['KNeighborsRegressor'            ] = (neighbors.KNeighborsRegressor            , 'float' ,  True )# Regression based on k-nearest neighbors.
  availImpl['neighbors']['RadiusNeighborsRegressor'       ] = (neighbors.RadiusNeighborsRegressor       , 'float' ,  True )# Regression based on neighbors within a fixed radius.
  availImpl['neighbors']['NearestCentroid'                ] = (neighbors.NearestCentroid                , 'int'   ,  True )# Nearest centroid classifier.
  availImpl['neighbors']['BallTree'                       ] = (neighbors.BallTree                       , 'float' ,  True )# BallTree for fast generalized N-point problems
  availImpl['neighbors']['KDTree'                         ] = (neighbors.KDTree                         , 'float' ,  True )# KDTree for fast generalized N-point problems

  availImpl['qda'] = {}
  availImpl['qda']['QDA'                                  ] = (qda.QDA                                  , 'int'   ,  False) #Quadratic Discriminant Analysis (QDA)
  # availImpl['qda']['QDA'                                  ] = (discriminant_analysis.QuadraticDiscriminantAnalysis, 'int'   ,  False) #Quadratic Discriminant Analysis (QDA)

  availImpl['tree'] = {}
  availImpl['tree']['DecisionTreeClassifier'              ] = (tree.DecisionTreeClassifier              , 'int'   ,  True )# A decision tree classifier.
  availImpl['tree']['DecisionTreeRegressor'               ] = (tree.DecisionTreeRegressor               , 'float' ,  True )# A tree regressor.
  availImpl['tree']['ExtraTreeClassifier'                 ] = (tree.ExtraTreeClassifier                 , 'int'   ,  True )# An extremely randomized tree classifier.
  availImpl['tree']['ExtraTreeRegressor'                  ] = (tree.ExtraTreeRegressor                  , 'float' ,  True )# An extremely randomized tree regressor.

  availImpl['GaussianProcess'] = {}
  availImpl['GaussianProcess']['GaussianProcess'          ] = (gaussian_process.GaussianProcess         , 'float' ,  False)
  #test if a method to estimate the probability of the prediction is available
  qualityEstTypeDict = {}
  for key1, myDict in availImpl.items():
    qualityEstTypeDict[key1] = {}
    for key2 in myDict:
      qualityEstTypeDict[key1][key2] = []
      if  callable(getattr(myDict[key2][0], "predict_proba", None))  : qualityEstTypeDict[key1][key2] += ['probability']
      elif  callable(getattr(myDict[key2][0], "score"        , None)): qualityEstTypeDict[key1][key2] += ['score']
      else                                                           : qualityEstTypeDict[key1][key2] = False

  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately initialize a supervised learning object
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    superVisedLearning.__init__(self,messageHandler,**kwargs)
    name  = self.initOptionDict.pop('name','')
    self.printTag = 'SCIKITLEARN'
    if 'SKLtype' not in self.initOptionDict.keys():
      self.raiseAnError(IOError,'to define a scikit learn ROM the SKLtype keyword is needed (from ROM "'+name+'")')
    SKLtype, SKLsubType = self.initOptionDict['SKLtype'].split('|')
    self.subType = SKLsubType
    self.intrinsicMultiTarget     = 'MultiTask' in self.initOptionDict['SKLtype']
    self.initOptionDict.pop('SKLtype')
    if not SKLtype in self.__class__.availImpl.keys():
      self.raiseAnError(IOError,'not known SKLtype "' + SKLtype +'" (from ROM "'+name+'")')
    if not SKLsubType in self.__class__.availImpl[SKLtype].keys():
      self.raiseAnError(IOError,'not known SKLsubType "' + SKLsubType +'" (from ROM "'+name+'")')

    self.__class__.returnType     = self.__class__.availImpl[SKLtype][SKLsubType][1]
    self.externalNorm             = self.__class__.availImpl[SKLtype][SKLsubType][2]
    self.__class__.qualityEstType = self.__class__.qualityEstTypeDict[SKLtype][SKLsubType]

    if 'estimator' in self.initOptionDict.keys():
      estimatorDict = self.initOptionDict['estimator']
      self.initOptionDict.pop('estimator')
      estimatorSKLtype, estimatorSKLsubType = estimatorDict['SKLtype'].split('|')
      estimator = self.__class__.availImpl[estimatorSKLtype][estimatorSKLsubType][0]()
      if self.intrinsicMultiTarget: self.ROM = [self.__class__.availImpl[SKLtype][SKLsubType][0](estimator)]
      else                        : self.ROM = [self.__class__.availImpl[SKLtype][SKLsubType][0](estimator) for _ in range(len(self.target))]
    else:
      if self.intrinsicMultiTarget: self.ROM = [self.__class__.availImpl[SKLtype][SKLsubType][0]()]
      else                        : self.ROM = [self.__class__.availImpl[SKLtype][SKLsubType][0]() for _ in range(len(self.target))]

    for key,value in self.initOptionDict.items():
      try   : self.initOptionDict[key] = ast.literal_eval(value)
      except: pass

    for index in range(len(self.ROM)): self.ROM[index].set_params(**self.initOptionDict)

  def _readdressEvaluateConstResponse(self,edict):
    """
      Method to re-address the evaluate base class method in order to avoid wasting time
      in case the training set has an unique response (e.g. if 10 points in the training set,
      and the 10 outcomes are all == to 1, this method returns one without the need of an
      evaluation)
      @ In, edict, dict, prediction request. Not used in this method (kept the consistency with evaluate method)
      @ Out, returnDict, dict, dictionary with the evaluation (in this case, the constant number)
    """
    returnDict = {}
    for index,target in enumerate(self.target): returnDict[target] = self.myNumber[index]
    return returnDict

  def _readdressEvaluateRomResponse(self,edict):
    """
      Method to re-address the evaluate base class method to its original method
      @ In, edict, dict, prediction request. Not used in this method (kept the consistency with evaluate method)
      @ Out, evaluate, float, the evaluation
    """
    return self.__class__.evaluate(self,edict)

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on samples in featureVals with responses y.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature values
      @ Out, targetVals, array, shape = [n_samples,n_targets], an array of output target
        associated with the corresponding points in featureVals
    """
    #If all the target values are the same no training is needed and the moreover the self.evaluate could be re-addressed to this value
    if self.intrinsicMultiTarget:
      self.ROM[0].fit(featureVals,targetVals)
    else:
      if not all([len(np.unique(targetVals[:,index]))>1 for index in range(len(self.ROM))]):
        self.myNumber = [np.unique(targetVals[:,index])[0] for index in range(len(self.ROM)) ]
        self.evaluate = self._readdressEvaluateConstResponse
      else:
        for index in range(len(self.ROM)):
          self.ROM[index].fit(featureVals,targetVals[:,index])
        self.evaluate = self._readdressEvaluateRomResponse

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals, 2-D numpy array, [n_samples,n_features]
      @ Out, confidenceDict, dict, dict of the dictionary for each target
    """
    confidenceDict = {}
    if  'probability' in self.__class__.qualityEstType:
      for index, target in enumerate(self.ROM):
        confidenceDict[target] =  self.ROM[index].predict_proba(featureVals)
    else            : self.raiseAnError(IOError,'the ROM '+str(self.initOptionDict['name'])+'has not the an method to evaluate the confidence of the prediction')
    return confidenceDict

  def __evaluateLocal__(self,featureVals):
    """
      Evaluates a point.
      @ In, featureVals, np.array, list of values at which to evaluate the ROM
      @ Out, returnDict, dict, dict of all the target results
    """
    returnDict = {}
    if not self.intrinsicMultiTarget:
      for index, target in enumerate(self.target): returnDict[target] = self.ROM[index].predict(featureVals)
    else:
      outcome = self.ROM[0].predict(featureVals)
      for index, target in enumerate(self.target): returnDict[target] = outcome[:,index]
    return returnDict

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    for index in range(len(self.ROM)):
      self.ROM[index] = self.ROM[index].__class__(**self.initOptionDict)

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    params = self.ROM[-1].get_params()
    return params

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    self.raiseADebug('here we need to collect some info on the ROM status')
    params = {}
    return params

  def _localNormalizeData(self,values,names,feat):
    """
      Overwrites default normalization procedure.
      @ In, values, list(float), unused
      @ In, names, list(string), unused
      @ In, feat, string, feature to (not) normalize
      @ Out, None
    """
    if not self.externalNorm:
      self.muAndSigmaFeatures[feat] = (0.0,1.0)
    else:
      super(SciKitLearn, self)._localNormalizeData(values,names,feat)
#
#
#

class ARMA(superVisedLearning):
  """
    Autoregressive Moving Average model for time series analysis. First train then evaluate.
    Specify a Fourier node in input file if detrending by Fourier series is needed.

    Time series Y: Y = X + \sum_{i}\sum_k [\delta_ki1*sin(2pi*k/basePeriod_i)+\delta_ki2*cos(2pi*k/basePeriod_i)]
    ARMA series X: x_t = \sum_{i=1}^P \phi_i*x_{t-i} + \alpha_t + \sum_{j=1}^Q \theta_j*\alpha_{t-j}
  """
  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler: a MessageHandler object in charge of raising errors,
                           and printing messages
      @ In, kwargs: an arbitrary dictionary of keywords and values
    """
    superVisedLearning.__init__(self,messageHandler,**kwargs)
    self.printTag          = 'ARMA'
    self._dynamicHandling  = True                                    # This ROM is able to manage the time-series on its own. No need for special treatment outside
    self.armaPara          = {}
    self.armaPara['Pmax']      = kwargs.get('Pmax', 3)
    self.armaPara['Pmin']      = kwargs.get('Pmin', 0)
    self.armaPara['Qmax']      = kwargs.get('Qmax', 3)
    self.armaPara['Qmin']      = kwargs.get('Qmin', 0)
    self.armaPara['dimension'] = len(self.features)
    self.outTruncation         = kwargs.get('outTruncation', None)     # Additional parameters to allow user to specify the time series to be all positive or all negative
    self.pivotParameterID      = kwargs.get('pivotParameter', 'Time')
    self.pivotParameterValues  = None                                  # In here we store the values of the pivot parameter (e.g. Time)
    # check if the pivotParameter is among the targetValues
    if self.pivotParameterID not in self.target: self.raiseAnError(IOError,"The pivotParameter "+self.pivotParameterID+" must be part of the Target space!")
    if len(self.target) > 2: self.raiseAnError(IOError,"Multi-target ARMA not available yet!")
    # Initialize parameters for Fourier detrending
    if 'Fourier' not in self.initOptionDict.keys():
      self.hasFourierSeries = False
    else:
      self.hasFourierSeries = True
      self.fourierPara = {}
      self.fourierPara['basePeriod'] = [float(temp) for temp in self.initOptionDict['Fourier'].split(',')]
      self.fourierPara['FourierOrder'] = {}
      if 'FourierOrder' not in self.initOptionDict.keys():
        for basePeriod in self.fourierPara['basePeriod']:
          self.fourierPara['FourierOrder'][basePeriod] = 4
      else:
        temps = self.initOptionDict['FourierOrder'].split(',')
        for index, basePeriod in enumerate(self.fourierPara['basePeriod']):
          self.fourierPara['FourierOrder'][basePeriod] = int(temps[index])
      if len(self.fourierPara['basePeriod']) != len(self.fourierPara['FourierOrder']):
        self.raiseAnError(ValueError, 'Length of FourierOrder should be ' + str(len(self.fourierPara['basePeriod'])))

  def _localNormalizeData(self,values,names,feat): # This function is not used in this class and can be removed
    """
      Overwrites default normalization procedure.
      @ In, values, unused
      @ In, names, unused
      @ In, feat, feature to normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def __trainLocal__(self,featureVals,targetVals):
    """
      Perform training on input database stored in featureVals.

      @ In, featureVals, array, shape=[n_timeStep, n_dimensions], an array of input data # Not use for ARMA training
      @ In, targetVals, array, shape = [n_timeStep, n_dimensions], an array of time series data
    """
    self.pivotParameterValues = targetVals[:,:,self.target.index(self.pivotParameterID)]
    if len(self.pivotParameterValues) > 1: self.raiseAnError(Exception,self.printTag +" does not handle multiple histories data yet! # histories: "+str(len(self.pivotParameterValues)))
    self.pivotParameterValues.shape = (self.pivotParameterValues.size,)
    self.timeSeriesDatabase         = copy.deepcopy(np.delete(targetVals,self.target.index(self.pivotParameterID),2))
    self.timeSeriesDatabase.shape   = (self.timeSeriesDatabase.size,)
    self.target.pop(self.target.index(self.pivotParameterID))
    # Fit fourier seires
    if self.hasFourierSeries:
      self.__trainFourier__()
      self.armaPara['rSeries'] = self.timeSeriesDatabase - self.fourierResult['predict']
    else:
      self.armaPara['rSeries'] = self.timeSeriesDatabase

#     Transform data to obatain normal distrbuted series. See
#     J.M.Morales, R.Minguez, A.J.Conejo "A methodology to generate statistically dependent wind speed scenarios,"
#     Applied Energy, 87(2010) 843-855
    self.__generateCDF__(self.armaPara['rSeries'])
    self.armaPara['rSeriesNorm'] = self.__dataConversion__(self.armaPara['rSeries'], obj='normalize')

    self.__trainARMA__() # Fit ARMA model: x_t = \sum_{i=1}^P \phi_i*x_{t-i} + \alpha_t + \sum_{j=1}^Q \theta_j*\alpha_{t-j}

    del self.timeSeriesDatabase       # Delete to reduce the pickle size, since from now the original data will no longer be used in the evaluation.

  def __trainFourier__(self):
    """
      Perform fitting of Fourier series on self.timeSeriesDatabase
      @ In, none,
      @ Out, none,
    """
    fourierSeriesAll = self.__generateFourierSignal__(self.pivotParameterValues, self.fourierPara['basePeriod'], self.fourierPara['FourierOrder'])
    fourierEngine = linear_model.LinearRegression()
    temp = {}
    for bp in self.fourierPara['FourierOrder'].keys():
      temp[bp] = range(1,self.fourierPara['FourierOrder'][bp]+1)
    fourOrders = list(itertools.product(*temp.values())) # generate the set of combinations of the Fourier order

    criterionBest = np.inf
    fSeriesBest = []
    self.fourierResult={}
    self.fourierResult['residues'] = 0
    self.fourierResult['fOrder'] = []

    for fOrder in fourOrders:
      fSeries = np.zeros(shape=(self.pivotParameterValues.size,2*sum(fOrder)))
      indexTemp = 0
      for index,bp in enumerate(self.fourierPara['FourierOrder'].keys()):
        fSeries[:,indexTemp:indexTemp+fOrder[index]*2] = fourierSeriesAll[bp][:,0:fOrder[index]*2]
        indexTemp += fOrder[index]*2
      fourierEngine.fit(fSeries,self.timeSeriesDatabase)
      r = (fourierEngine.predict(fSeries)-self.timeSeriesDatabase)**2
      if r.size > 1:    r = sum(r)
      r = r/self.pivotParameterValues.size
      criterionCurrent = copy.copy(r)
      if  criterionCurrent< criterionBest:
        self.fourierResult['fOrder'] = copy.deepcopy(fOrder)
        fSeriesBest = copy.deepcopy(fSeries)
        self.fourierResult['residues'] = copy.deepcopy(r)
        criterionBest = copy.deepcopy(criterionCurrent)

    fourierEngine.fit(fSeriesBest,self.timeSeriesDatabase)
    self.fourierResult['predict'] = np.asarray(fourierEngine.predict(fSeriesBest))

  def __trainARMA__(self):
    """
      Fit ARMA model: x_t = \sum_{i=1}^P \phi_i*x_{t-i} + \alpha_t + \sum_{j=1}^Q \theta_j*\alpha_{t-j}
      Data series to this function has been normalized so that it is standard gaussian
      @ In, none,
      @ Out, none,
    """
    self.armaResult = {}
    Pmax = self.armaPara['Pmax']
    Pmin = self.armaPara['Pmin']
    Qmax = self.armaPara['Qmax']
    Qmin = self.armaPara['Qmin']

    criterionBest = np.inf
    for p in range(Pmin,Pmax+1):
      for q in range(Qmin,Qmax+1):
        if p is 0 and q is 0:     continue          # dump case so we pass
        init = [0.0]*(p+q)*self.armaPara['dimension']**2
        init_S = np.identity(self.armaPara['dimension'])
        for n1 in range(self.armaPara['dimension']): init.append(init_S[n1,n1])

        rOpt = {}
        rOpt = optimize.fmin(self.__computeARMALikelihood__,init, args=(p,q) ,full_output = True)
        tmp = (p+q)*self.armaPara['dimension']**2/self.pivotParameterValues.size
        criterionCurrent = self.__computeAICorBIC(self.armaResult['sigHat'],noPara=tmp,cType='BIC',obj='min')
        if criterionCurrent < criterionBest or 'P' not in self.armaResult.keys(): # to save the first iteration results
          self.armaResult['P'] = p
          self.armaResult['Q'] = q
          self.armaResult['param'] = rOpt[0]
          criterionBest = criterionCurrent

    # saving training results
    Phi, Theta, Cov = self.__armaParamAssemb__(self.armaResult['param'],self.armaResult['P'],self.armaResult['Q'],self.armaPara['dimension'] )
    self.armaResult['Phi'] = Phi
    self.armaResult['Theta'] = Theta
    self.armaResult['sig'] = np.zeros(shape=(1, self.armaPara['dimension'] ))
    for n in range(self.armaPara['dimension'] ):      self.armaResult['sig'][0,n] = np.sqrt(Cov[n,n])

  def __generateCDF__(self, data):
    """
      Generate empirical CDF function of the input data, and save the results in self
      @ In, data, array, shape = [n_timeSteps, n_dimension], data over which the CDF will be generated
      @ Out, none,
    """
    self.armaNormPara = {}
    self.armaNormPara['resCDF'] = {}

    if len(data.shape) == 1: data = np.reshape(data, newshape = (data.shape[0],1))
    num_bins = [0]*data.shape[1] # initialize num_bins, which will be calculated later by Freedman Diacoins rule

    for d in range(data.shape[1]):
      num_bins[d] = self.__computeNumberBins__(data[:,d])
      counts, binEdges = np.histogram(data[:,d], bins = num_bins[d], normed = True)
      Delta = np.zeros(shape=(num_bins[d],1))
      for n in range(num_bins[d]):      Delta[n,0] = binEdges[n+1]-binEdges[n]
      temp = np.cumsum(counts)*average(Delta)
      cdf = np.insert(temp, 0, temp[0]) # minimum of CDF is set to temp[0] instead of 0 to avoid numerical issues
      self.armaNormPara['resCDF'][d] = {}
      self.armaNormPara['resCDF'][d]['bins'] = copy.deepcopy(binEdges)
      self.armaNormPara['resCDF'][d]['binsMax'] = max(binEdges)
      self.armaNormPara['resCDF'][d]['binsMin'] = min(binEdges)
      self.armaNormPara['resCDF'][d]['CDF'] = copy.deepcopy(cdf)
      self.armaNormPara['resCDF'][d]['CDFMax'] = max(cdf)
      self.armaNormPara['resCDF'][d]['CDFMin'] = min(cdf)
      self.armaNormPara['resCDF'][d]['binSearchEng'] = neighbors.NearestNeighbors(n_neighbors=2).fit([[b] for b in binEdges])
      self.armaNormPara['resCDF'][d]['cdfSearchEng'] = neighbors.NearestNeighbors(n_neighbors=2).fit([[c] for c in cdf])

  def __computeNumberBins__(self, data):
    """
      Compute number of bins determined by Freedman Diaconis rule
      https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
      @ In, data, array, shape = [n_sample], data over which the number of bins is decided
      @ Out, numBin, int, number of bins determined by Freedman Diaconis rule
    """
    IQR = np.percentile(data, 75) - np.percentile(data, 25)
    binSize = 2.0*IQR*(data.size**(-1.0/3.0))
    numBin = int((max(data)-min(data))/binSize)
    return numBin

  def __getCDF__(self,d,x):
    """
      Get residue CDF value at point x for d-th dimension
      @ In, d, int, dimension id
      @ In, x, float, variable value for which the CDF is computed
      @ Out, y, float, CDF value
    """
    if x <= self.armaNormPara['resCDF'][d]['binsMin']:    y = self.armaNormPara['resCDF'][d]['CDF'][0]
    elif x >= self.armaNormPara['resCDF'][d]['binsMax']:  y = self.armaNormPara['resCDF'][d]['CDF'][-1]
    else:
      ind = self.armaNormPara['resCDF'][d]['binSearchEng'].kneighbors(x, return_distance=False)
      X, Y = self.armaNormPara['resCDF'][d]['bins'][ind], self.armaNormPara['resCDF'][d]['CDF'][ind]
      if X[0,0] <= X[0,1]:        x1, x2, y1, y2 = X[0,0], X[0,1], Y[0,0], Y[0,1]
      else:                       x1, x2, y1, y2 = X[0,1], X[0,0], Y[0,1], Y[0,0]
      if x1 == x2:                y = (y1+y2)/2.0
      else:                       y = y1 + 1.0*(y2-y1)/(x2-x1)*(x-x1)
    return y

  def __getInvCDF__(self,d,x):
    """
      Get inverse residue CDF at point x for d-th dimension
      @ In, d, int, dimension id
      @ In, x, float, the CDF value for which the inverse value is computed
      @ Out, y, float, variable value
    """
    if x < 0 or x > 1:    self.raiseAnError(ValueError, 'Input to __getRInvCDF__ is not in unit interval' )
    elif x <= self.armaNormPara['resCDF'][d]['CDFMin']:   y = self.armaNormPara['resCDF'][d]['bins'][0]
    elif x >= self.armaNormPara['resCDF'][d]['CDFMax']:   y = self.armaNormPara['resCDF'][d]['bins'][-1]
    else:
      ind = self.armaNormPara['resCDF'][d]['cdfSearchEng'].kneighbors(x, return_distance=False)
      X, Y = self.armaNormPara['resCDF'][d]['CDF'][ind], self.armaNormPara['resCDF'][d]['bins'][ind]
      if X[0,0] <= X[0,1]:        x1, x2, y1, y2 = X[0,0], X[0,1], Y[0,0], Y[0,1]
      else:                       x1, x2, y1, y2 = X[0,1], X[0,0], Y[0,1], Y[0,0]
      if x1 == x2:                y = (y1+y2)/2.0
      else:                       y = y1 + 1.0*(y2-y1)/(x2-x1)*(x-x1)
    return y

  def __dataConversion__(self, data, obj):
    """
      Transform input data to a Normal/empirical distribution data set.
      @ In, data, array, shape=[n_timeStep, n_dimension], input data to be transformed
      @ In, obj, string, specify whether to normalize or denormalize the data
      @ Out, transformedData, array, shape = [n_timeStep, n_dimension], output transformed data that has normal/empirical distribution
    """
    # Instantiate a normal distribution for data conversion
    normTransEngine = Distributions.returnInstance('Normal',self)
    normTransEngine.mean, normTransEngine.sigma = 0, 1
    normTransEngine.upperBoundUsed, normTransEngine.lowerBoundUsed = False, False
    normTransEngine.initializeDistribution()

    if len(data.shape) == 1: data = np.reshape(data, newshape = (data.shape[0],1))
    # Transform data
    transformedData = np.zeros(shape=data.shape)
    for n1 in range(data.shape[0]):
      for n2 in range(data.shape[1]):
        if obj in ['normalize']:
          temp = self.__getCDF__(n2, data[n1,n2])
          # for numerical issues, value less than 1 returned by __getCDF__ can be greater than 1 when stored in temp
          # This might be a numerical issue of dependent library.
          # It seems gone now. Need further investigation.
          if temp >= 1:                temp = 1 - np.finfo(float).eps
          elif temp <= 0:              temp = np.finfo(float).eps
          transformedData[n1,n2] = normTransEngine.ppf(temp)
        elif obj in ['denormalize']:
          temp = normTransEngine.cdf(data[n1, n2])
          transformedData[n1,n2] = self.__getInvCDF__(n2, temp)
        else:       self.raiseAnError(ValueError, 'Input obj to __dataConversion__ is not properly set')
    return transformedData

  def __generateFourierSignal__(self, Time, basePeriod, fourierOrder):
    """
      Generate fourier signal as specified by the input file
      @ In, basePeriod, list, list of base periods
      @ In, fourierOrder, dict, order for each base period
      @ Out, fourierSeriesAll, array, shape = [n_timeStep, n_basePeriod]
    """
    fourierSeriesAll = {}
    for bp in basePeriod:
      fourierSeriesAll[bp] = np.zeros(shape=(Time.size, 2*fourierOrder[bp]))
      for orderBp in range(fourierOrder[bp]):
        fourierSeriesAll[bp][:, 2*orderBp] = np.sin(2*np.pi*(orderBp+1)/bp*Time)
        fourierSeriesAll[bp][:, 2*orderBp+1] = np.cos(2*np.pi*(orderBp+1)/bp*Time)
    return fourierSeriesAll

  def __armaParamAssemb__(self,x,p,q,N):
    """
      Assemble ARMA parameter into matrices
      @ In, x, list, ARMA parameter stored as vector
      @ In, p, int, AR order
      @ In, q, int, MA order
      @ In, N, int, dimensionality of x
      @ Out Phi, list, list of Phi parameters (each as an array) for each AR order
      @ Out Theta, list, list of Theta parameters (each as an array) for each MA order
      @ Out Cov, array, covariance matrix of the noise
    """
    Phi, Theta, Cov = {}, {}, np.identity(N)
    for i in range(1,p+1):
      Phi[i] = np.zeros(shape=(N,N))
      for n in range(N):      Phi[i][n,:] = x[N**2*(i-1)+n*N:N**2*(i-1)+(n+1)*N]
    for j in range(1,q+1):
      Theta[j] = np.zeros(shape=(N,N))
      for n in range(N):      Theta[j][n,:] = x[N**2*(p+j-1)+n*N:N**2*(p+j-1)+(n+1)*N]
    for n in range(N):        Cov[n,n] = x[N**2*(p+q)+n]
    return Phi, Theta, Cov

  def __computeARMALikelihood__(self,x,*args):
    """
      Compute the likelihood given an ARMA model
      @ In, x, list, ARMA parameter stored as vector
      @ In, args, dict, additional argument
      @ Out, lkHood, float, output likelihood
    """
    if len(args) != 2:    self.raiseAnError(ValueError, 'args to __computeARMALikelihood__ should have exactly 2 elements')

    p, q, N = args[0], args[1], self.armaPara['dimension']
    if len(x) != N**2*(p+q)+N:    self.raiseAnError(ValueError, 'input to __computeARMALikelihood__ has wrong dimension')
    Phi, Theta, Cov = self.__armaParamAssemb__(x,p,q,N)
    for n1 in range(N):
      for n2 in range(N):
        if Cov[n1,n2] <0:
          lkHood = sys.float_info.max
          return lkHood

    CovInv = np.linalg.inv(Cov)
    d = self.armaPara['rSeriesNorm']
    numTimeStep = d.shape[0]
    alpha = np.zeros(shape=d.shape)
    L = -N*numTimeStep/2.0*np.log(2*np.pi) - numTimeStep/2.0*np.log(np.linalg.det(Cov))
    for t in range(numTimeStep):
      alpha[t,:] = d[t,:]
      for i in range(1,min(p,t)+1):     alpha[t,:] -= np.dot(Phi[i],d[t-i,:])
      for j in range(1,min(q,t)+1):     alpha[t,:] -= np.dot(Theta[j],alpha[t-j,:])
      L -= 1/2.0*np.dot(np.dot(alpha[t,:].T,CovInv),alpha[t,:])

    sigHat = np.dot(alpha.T,alpha)
    while sigHat.size > 1:
      sigHat = sum(sigHat)
      sigHat = sum(sigHat.T)
    sigHat = sigHat / numTimeStep
    self.armaResult['sigHat'] = sigHat[0,0]
    lkHood = -L
    return lkHood

  def __computeAICorBIC(self,maxL,noPara,cType,obj='max'):
    """
      Compute the AIC or BIC criteria for model selection.
      @ In, maxL, float, likelihood of given parameters
      @ In, noPara, int, number of parameters
      @ In, cType, string, specify whether AIC or BIC should be returned
      @ In, obj, string, specify the optimization is for maximum or minimum.
      @ Out, criterionValue, float, value of AIC/BIC
    """
    if obj == 'min':        flag = -1
    else:                   flag = 1
    if cType == 'BIC':      criterionValue = -1*flag*np.log(maxL)+noPara*np.log(self.pivotParameterValues.size)
    elif cType == 'AIC':    criterionValue = -1*flag*np.log(maxL)+noPara*2
    else:                   criterionValue = maxL
    return criterionValue

  def __evaluateLocal__(self,featureVals):
    """
      @ In, featureVals, float, a scalar feature value is passed as scaling factor
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """
    if featureVals.size > 1:
      self.raiseAnError(ValueError, 'The input feature for ARMA for evaluation cannot have size greater than 1. ')

    # Instantiate a normal distribution for time series synthesis (noise part)
    normEvaluateEngine = Distributions.returnInstance('Normal',self)
    normEvaluateEngine.mean, normEvaluateEngine.sigma = 0, 1
    normEvaluateEngine.upperBoundUsed, normEvaluateEngine.lowerBoundUsed = False, False
    normEvaluateEngine.initializeDistribution()

    numTimeStep = len(self.pivotParameterValues)
    tSeriesNoise = np.zeros(shape=self.armaPara['rSeriesNorm'].shape)
    for t in range(numTimeStep):
      for n in range(self.armaPara['dimension']):
        tSeriesNoise[t,n] = normEvaluateEngine.rvs()*self.armaResult['sig'][0,n]

    tSeriesNorm = np.zeros(shape=(numTimeStep,self.armaPara['rSeriesNorm'].shape[1]))
    tSeriesNorm[0,:] = self.armaPara['rSeriesNorm'][0,:]
    for t in range(numTimeStep):
      for i in range(1,min(self.armaResult['P'], t)+1):
        tSeriesNorm[t,:] += np.dot(tSeriesNorm[t-i,:], self.armaResult['Phi'][i])
      for j in range(1,min(self.armaResult['Q'], t)+1):
        tSeriesNorm[t,:] += np.dot(tSeriesNoise[t-j,:], self.armaResult['Theta'][j])
      tSeriesNorm[t,:] += tSeriesNoise[t,:]

    # Convert data back to empirically distributed
    tSeries = self.__dataConversion__(tSeriesNorm, obj='denormalize')
    # Add fourier trends
    self.raiseADebug(self.fourierResult['predict'].shape, tSeries.shape)
    if self.hasFourierSeries:
      if len(self.fourierResult['predict'].shape) == 1:
        tempFour = np.reshape(self.fourierResult['predict'], newshape=(self.fourierResult['predict'].shape[0],1))
      else:
        tempFour = self.fourierResult['predict'][0:numTimeStep,:]
      tSeries += tempFour
    # Ensure positivity --- FIXME
    if self.outTruncation is not None:
      if self.outTruncation == 'positive':      tSeries = np.absolute(tSeries)
      elif self.outTruncation == 'negative':    tSeries = -np.absolute(tSeries)
    returnEvaluation = {}
    returnEvaluation[self.pivotParameterID] = self.pivotParameterValues[0:numTimeStep]
    evaluation = tSeries*featureVals
    for index, target in enumerate(self.target): returnEvaluation[target] = evaluation[:,index]
    return returnEvaluation

  def __confidenceLocal__(self,featureVals):
    """
      This method is currently not needed for ARMA
    """
    pass

  def __resetLocal__(self,featureVals):
    """
      After this method the ROM should be described only by the initial parameter settings
      Currently not implemented for ARMA
    """
    pass

  def __returnInitialParametersLocal__(self):
    """
      there are no possible default parameters to report
    """
    localInitParam = {}
    return localInitParam

  def __returnCurrentSettingLocal__(self):
    """
      override this method to pass the set of parameters of the ROM that can change during simulation
      Currently not implemented for ARMA
    """
    pass

__interfaceDict                         = {}
__interfaceDict['NDspline'            ] = NDsplineRom
__interfaceDict['NDinvDistWeight'     ] = NDinvDistWeight
__interfaceDict['SciKitLearn'         ] = SciKitLearn
__interfaceDict['GaussPolynomialRom'  ] = GaussPolynomialRom
__interfaceDict['HDMRRom'             ] = HDMRRom
__interfaceDict['MSR'                 ] = MSR
__interfaceDict['ARMA'                ] = ARMA
__interfaceDict['pickledROM'          ] = pickledROM
__base                                  = 'superVisedLearning'

def returnStaticCharacteristics(infoType,ROMclass,caller,**kwargs):
  """
    This method is aimed to get the static characteristics of a certain ROM (e.g. multi-target, dynamic, etc.)
  """


def returnInstance(ROMclass,caller,**kwargs):
  """
    This function return an instance of the request model type
    @ In, ROMclass, string, string representing the instance to create
    @ In, caller, instance, object that will share its messageHandler instance
    @ In, kwargs, dict, a dictionary specifying the keywords and values needed to create the instance.
    @ Out, returnInstance, instance, an instance of a ROM
  """
  try: return __interfaceDict[ROMclass](caller.messageHandler,**kwargs)
  except KeyError as ae: caller.raiseAnError(NameError,'not known '+__base+' type '+str(ROMclass))

def returnClass(ROMclass,caller):
  """
    This function return an instance of the request model type
    @ In, ROMclass, string, string representing the class to retrieve
    @ In, caller, instnace, object that will share its messageHandler instance
    @ Out, returnClass, the class definition of a ROM
  """
  try: return __interfaceDict[ROMclass]
  except KeyError: caller.raiseAnError(NameError,'not known '+__base+' type '+ROMclass)
