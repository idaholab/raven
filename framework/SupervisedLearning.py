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
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
from sklearn import linear_model
from sklearn import svm
from sklearn import multiclass
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import qda
from sklearn import tree
from sklearn import lda
from sklearn import gaussian_process
import numpy as np
import abc
import ast
from operator import itemgetter
import math
from scipy import spatial
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
import mathUtils
import MessageHandler
import TreeStructure
interpolationND = utils.find_interpolationND()
#Internal Modules End--------------------------------------------------------------------------------

class superVisedLearning(utils.metaclass_insert(abc.ABCMeta),MessageHandler.MessageUser):
  """
    This is the general interface to any superVisedLearning learning method.
    Essentially it contains a train method and an evaluate method
  """
  returnType      = '' #this describe the type of information generated the possibility are 'boolean', 'integer', 'float'
  qualityEstType  = [] #this describe the type of estimator returned known type are 'distance', 'probability'. The values are returned by the self.__confidenceLocal__(Features)
  ROMtype         = '' #the broad class of the interpolator

  @staticmethod
  def checkArrayConsistency(arrayIn):
    """
      This method checks the consistency of the in-array
      @ In, arrayIn, object,  It should be an array
      @ Out, (consistent, 'error msg'), tuple, tuple[0] is a bool (True -> everything is ok, False -> something wrong), tuple[1], string ,the error mesg
    """
    #checking if None provides a more clear message about the problem
    if arrayIn is None: return (False,' The object is None, and contains no entries!')
    if type(arrayIn) != np.ndarray: return (False,' The object is not a numpy array')
    if len(arrayIn.shape) > 1: return(False, ' The array must be 1-d')
    return (True,'')

  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    self.printTag = 'Supervised'
    self.messageHandler = messageHandler
    #booleanFlag that controls the normalization procedure. If true, the normalization is performed. Default = True
    if kwargs != None: self.initOptionDict = kwargs
    else             : self.initOptionDict = {}
    if 'Features' not in self.initOptionDict.keys(): self.raiseAnError(IOError,'Feature names not provided')
    if 'Target'   not in self.initOptionDict.keys(): self.raiseAnError(IOError,'Target name not provided')
    self.features = self.initOptionDict['Features'].split(',')
    self.target   = self.initOptionDict['Target'  ]
    self.initOptionDict.pop('Target')
    self.initOptionDict.pop('Features')
    self.verbosity = self.initOptionDict['verbosity'] if 'verbosity' in self.initOptionDict else None
    if self.features.count(self.target) > 0: self.raiseAnError(IOError,'The target and one of the features have the same name!')
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
    if isinstance(self.target,list):
      targetValues = None
      for target in self.target:
        if target in names:
          if targetValues is None:
            targetValues = values[names.index(target)]
          else:
            targetValues = np.column_stack((targetValues,values[names.index(target)]))
        else:
          self.raiseAnError(IOError,'The target '+target+' is not in the training set')
      # construct the evaluation matrixes
      featureValues = np.zeros(shape=(len(targetValues),len(self.features)))
    else:
      if self.target in names:
        targetValues = values[names.index(self.target)]
      else:
        self.raiseAnError(IOError,'The target ' + self.target + ' is not in the training set')
      # check if the targetValues are consistent with the expected structure
      resp = self.checkArrayConsistency(targetValues)
      if not resp[0]:
        self.raiseAnError(IOError,'In training set for target '+self.target+':'+resp[1])
      # construct the evaluation matrixes
      featureValues = np.zeros(shape=(targetValues.size,len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names:
        self.raiseAnError(IOError,'The feature sought '+feat+' is not in the training set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]:
          self.raiseAnError(IOError,'In training set for feature '+feat+':'+resp[1])
        if values[names.index(feat)].size != featureValues[:,0].size:
          self.raiseAWarning('feature values:',featureValues[:,0].size,tag='ERROR')
          self.raiseAWarning('target values:',values[names.index(feat)].size,tag='ERROR')
          self.raiseAnError(IOError,'In training set, the number of values provided for feature '+feat+' are != number of target outcomes!')
        self._localNormalizeData(values,names,feat)
        if self.muAndSigmaFeatures[feat][1]==0:
          self.muAndSigmaFeatures[feat] = (self.muAndSigmaFeatures[feat][0],np.max(np.absolute(values[names.index(feat)])))
        if self.muAndSigmaFeatures[feat][1]==0:
          self.muAndSigmaFeatures[feat] = (self.muAndSigmaFeatures[feat][0],1.0)
        featureValues[:,cnt] = (values[names.index(feat)] - self.muAndSigmaFeatures[feat][0])/self.muAndSigmaFeatures[feat][1]
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
    self.muAndSigmaFeatures[feat] = (np.average(values[names.index(feat)]),np.std(values[names.index(feat)]))

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

  def printXML(self,rootnode,options=None):
    """
      Allows the SVE to put whatever it wants into an XML to print to file.
      @ In, rootnode, xml.etree.ElementTree.Element, the root node of an XML tree to print to
      @ In, options, dict, optional, dict of string-based options to use, including filename, things to print, etc
      @ Out, None
    """
    node = TreeStructure.Node(self.target)
    rootnode.appendBranch(node)
    self._localPrintXML(node,options)

  def _localPrintXML(self,node,options=None):
    """
      Specific local method for printing anything desired to xml file.  Overwrite in inheriting classes.
      @ In, node, the node to which strings should have text added
      @ In, options, dict of string-based options to use, including filename, things to print, etc
      @ Out, None
    """
    node.addText('ROM of type '+str(self.printTag.strip())+' has no special output options.')

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
    self.interpolator = None  # pointer to the C++ (crow) interpolator
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
    targv = interpolationND.vectd(targetVals)

    self.interpolator.fit(featv,targv)

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
    prediction = np.zeros(featureVals.shape[0])
    for n_sample in range(featureVals.shape[0]):
      featv = interpolationND.vectd(featureVals[n_sample][:])
      prediction[n_sample] = self.interpolator.interpolateAt(featv)
      self.raiseAMessage('NDinterpRom   : Prediction by ' + self.__class__.ROMtype + '. Predicted value is ' + str(prediction[n_sample]))
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
    self.printTag      = 'GAUSSgpcROM('+self.target+')'
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

  def _localPrintXML(self,node,options=None):
    """
      Adds requested entries to XML node.
      @ In, node, XML node, to which entries will be added
      @ In, options, dict, optional, list of requests and options
        May include:
        '  what': comma-separated string list, the qualities to print out
      @ Out, None
    """
    if not self.amITrained: self.raiseAnError(RuntimeError,'ROM is not yet trained!')
    self.mean=None
    canDo = ['mean','variance','samples','polyCoeffs','indices']
    if 'what' in options.keys():
      requests = list(o.strip() for o in options['what'].split(','))
      if 'all' in requests: requests = canDo
      for request in requests:
        request=request.strip()
        newNode = TreeStructure.Node(request)
        if request.lower() in ['mean','expectedvalue']:
          if self.mean == None: self.mean = self.__mean__()
          newNode.setText(self.mean)
        elif request.lower() in ['variance']:
          newNode.setText(self.__variance__())
        elif request.lower() in ['samples']:
          if self.numRuns!=None: newNode.setText(self.numRuns)
          else: newNode.setText(len(self.sparseGrid))
        elif request.lower() in ['polycoeffs']:
          vNode = TreeStructure.Node('inputVariables')
          vNode.text = ','.join(self.features)
          newNode.appendBranch(vNode)
          keys = self.polyCoeffDict.keys()
          keys.sort()
          for key in keys:
            cNode = TreeStructure.Node('_'+'_'.join(str(k) for k in key)+'_')
            cNode.setText(self.polyCoeffDict[key])
            newNode.appendBranch(cNode)
        elif request.lower() in ['indices']:
          indices,partials = self.getSensitivities()
          totals = self.getTotalSensitivities(indices)
          #provide variance
          varNode = TreeStructure.Node('tot_variance')
          varNode.setText(self.__variance__())
          newNode.appendBranch(varNode)
          #sort by value
          entries = []
          for key in indices.keys():
            entries.append( (','.join(key),partials[key],indices[key],totals[key]) )
          entries.sort(key=lambda x: abs(x[1]),reverse=True)
          #add to tree
          for entry in entries:
            subNode = TreeStructure.Node('variables')
            subNode.setText(entry[0])
            vNode = TreeStructure.Node('partial_variance')
            vNode.setText(entry[1])
            subNode.appendBranch(vNode)
            vNode = TreeStructure.Node('Sobol_index')
            vNode.setText(entry[2])
            subNode.appendBranch(vNode)
            vNode = TreeStructure.Node('Sobol_total_index')
            vNode.setText(entry[3])
            subNode.appendBranch(vNode)
            newNode.appendBranch(subNode)
        else:
          self.raiseAWarning('ROM does not know how to return '+request)
          newNode.setText('not found')
        node.appendBranch(newNode)

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
    self.polyCoeffDict={}
    #check equality of point space
    self.raiseADebug('...checking required points are available...')
    fvs = []
    tvs=[]
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
      #brute way
      #found,idx,point = mathUtils.NDInArray(featureVals,pt)
      #end brute way
      if found:
        fvs.append(point)
        tvs.append(targetVals[idx])
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
      self.polyCoeffDict[idx]=0
      wtsum=0
      for pt,soln in zip(fvs,tvs):
        tupPt = tuple(pt)
        stdPt = standardPoints[tupPt]
        wt = self.sparseGrid.weights(translate[tupPt])
        self.polyCoeffDict[idx]+=soln*self._multiDPolyBasisEval(idx,stdPt)*wt
      self.polyCoeffDict[idx]*=self.norm
    self.amITrained=True
    self.raiseADebug('...training complete!')

  def printPolyDict(self,printZeros=False):
    """
      Human-readable version of the polynomial chaos expansion.
      @ In, printZeros, bool, optional, optional flag for printing even zero coefficients
      @ Out, None
    """
    data=[]
    for idx,val in self.polyCoeffDict.items():
      if abs(val) > 1e-12 or printZeros:
        data.append([idx,val])
    data.sort()
    self.raiseADebug('polyDict for ['+self.target+'] with inputs '+str(self.features)+':')
    for idx,val in data:
      self.raiseADebug('    '+str(idx)+' '+str(val))

  def checkForNonzeros(self,tol=1e-12):
    """
      Checks poly coefficient dictionary for nonzero entries.
      @ In, tol, float, optional, the tolerance under which is zero (default 1e-12)
      @ Out, data, list(tuple), the indices and values of the nonzero coefficients
    """
    data=[]
    for idx,val in self.polyCoeffDict.items():
      if round(val,11) !=0:
        data.append([idx,val])
    return data

  def __mean__(self):
    """
      Returns the mean of the ROM.
      @ In, None
      @ Out, __mean__, float, the mean
    """
    return self.__evaluateMoment__(1)

  def __variance__(self):
    """
      returns the variance of the ROM.
      @ In, None
      @ Out, __variance__, float, variance
    """
    mean = self.__evaluateMoment__(1)
    return self.__evaluateMoment__(2) - mean*mean

  def __evaluateMoment__(self,r):
    """
      Use the ROM's built-in method to calculate moments.
      @ In r, int, moment to calculate
      @ Out, tot, float, evaluation of moment
    """
    #TODO is there a faster way still to do this?
    if r==1: return self.polyCoeffDict[tuple([0]*len(self.features))]
    elif r==2: return sum(s**2 for s in self.polyCoeffDict.values())
    tot=0
    for pt,wt in self.sparseGrid:
      tot+=self.__evaluateLocal__([pt])**r*wt
    tot*=self.norm
    return tot

  def __evaluateLocal__(self,featureVals):
    """
      Evaluates a point.
      @ In, featureVals, list, of values at which to evaluate the ROM
      @ Out, tot, float, the evaluated point
    """
    featureVals=featureVals[0]
    tot=0
    stdPt = np.zeros(len(featureVals))
    for p,pt in enumerate(featureVals):
      varName = self.sparseGrid.varNames[p]
      stdPt[p] = self.distDict[varName].convertToQuad(self.quads[varName].type,pt)
    for idx,coeff in self.polyCoeffDict.items():
      tot+=coeff*self._multiDPolyBasisEval(idx,stdPt)
    return tot

  def _printPolynomial(self):
    """
      Prints each polynomial for each coefficient.
      @ In, None
      @ Out, None
    """
    self.raiseADebug('Coeff Idx')
    for idx,coeff in self.polyCoeffDict.items():
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

  def getSensitivities(self):
    """
      Calculates the Sobol indices (percent partial variances) of the terms in this expansion.
      @ In, None
      @ Out, getSensitivities, tuple(dict), Sobol indices and partial variances keyed by subset
    """
    totVar = self.__variance__()
    partials = {}
    #calculate partial variances
    self.raiseADebug('Calculating partial variances...')
    for poly,coeff in self.polyCoeffDict.items():
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
    self.printTag      = 'HDMR_ROM('+self.target+')'
    self.sobolOrder    = None #depth of HDMR/Sobol expansion
    self.ROMs          = {}   #dict of GaussPolyROM objects keyed by combination of vars that make them up
    self.sdx           = None #dict of sobol sensitivity coeffs, keyed on order and tuple(varnames)
    self.mean          = None #mean, store to avoid recalculation
    self.variance      = None #variance, store to avoid recalculation
    self.anova         = None #converted true ANOVA terms, stores coefficients not polynomials
    self.partialVariances = None #partial variance contributions

    for key,val in kwargs.items():
      if key=='SobolOrder': self.sobolOrder = int(val)

  def _localPrintXML(self,node,options=None):
    """
      Adds requested entries to XML node.
      @ In, node, XML node, to which entries will be added
      @ In, options, dict, optional, list of requests and options
        May include:
          'what': comma-separated string list, the qualities to print out
      @ Out, None
    """
    #inherit from GaussPolynomialRom
    if not self.amITrained: self.raiseAnError(RuntimeError,'ROM is not yet trained!')
    self.mean=None
    canDo = ['mean','variance','samples','indices']
    if 'what' in options.keys():
      requests = list(o.strip() for o in options['what'].split(','))
      if 'all' in requests: requests = canDo
      #protect against things SCgPC can do that HDMR can't
      if 'polyCoeffs' in requests:
        self.raiseAWarning('HDMRRom cannot currently print polynomial coefficients.  Skipping...')
        requests.remove('polyCoeffs')
      options['what'] = ','.join(requests)
    GaussPolynomialRom._localPrintXML(self,node,options)

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
    for i in range(len(featureVals)):
      ft[tuple(featureVals[i])]=targetVals[i]
    #get the reference case
    self.refpt = tuple(self.__fillPointWithRef((),[]))
    self.refSoln = ft[self.refpt]
    for combo,rom in self.ROMs.items():
      subtdict={}
      for c in combo: subtdict[c]=[]
      subtdict[self.target]=[]
      SG = rom.sparseGrid
      fvals=np.zeros([len(SG),len(combo)])
      tvals=np.zeros(len(SG))
      for i in range(len(SG)):
        getpt=tuple(self.__fillPointWithRef(combo,SG[i][0]))
        #the 1e-10 is to be consistent with RAVEN's CSV print precision
        tvals[i] = ft[tuple(mathUtils.NDInArray(np.array(ft.keys()),getpt,tol=1e-10)[2])]
        for fp,fpt in enumerate(SG[i][0]):
          fvals[i][fp] = fpt
      for i,c in enumerate(combo):
        subtdict[c] = fvals[:,i]
      subtdict[self.target] = tvals
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
    if not self.amITrained: self.raiseAnError(IOError,'Cannot evaluate, as ROM is not trained!')
    tot = 0
    for term,mult in self.reducedTerms.items():
      if term == ():
        tot += self.refSoln
      else:
        cutVals = [list(featureVals[0][self.features.index(j)] for j in term)]
        tot += self.ROMs[term].__evaluateLocal__(cutVals)*mult
    return tot

  def __mean__(self):
    """
      The Cut-HDMR approximation can return its mean easily.
      @ In, None
      @ Out, __mean__, float, the mean
    """
    if not self.amITrained: self.raiseAnError(IOError,'Cannot evaluate mean, as ROM is not trained!')
    return self._calcMean(self.reducedTerms)

  def __variance__(self):
    """
      The Cut-HDMR approximation can return its variance somewhat easily.
      @ In, None
      @ Out, __variance__, float, the variance
    """
    if not self.amITrained: self.raiseAnError(IOError,'Cannot evaluate variance, as ROM is not trained!')
    self.getSensitivities()
    return sum(val for val in self.partialVariances.values())

  def _calcMean(self,fromDict):
    """
      Given a subset, calculate mean from terms
      @ In, fromDict, dict{string:int}, ROM subsets and their multiplicity
      @ Out, tot, float, mean
    """
    tot = 0
    for term,mult in fromDict.items():
      tot += self._evaluateIntegral(term)*mult
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

  def _evaluateIntegral(self,term):
    """
      Uses properties of orthonormal gPC to algebraically evaluate integrals gPC
      This does assume the integral is over all the constituent variables in the the term
      @ In, term, string, subset term to integrate
      @ Out, _evaluateIntegral, float, evaluation

    """
    if term in [(),'',None]:
      return self.refSoln
    else:
      return self.ROMs[term].__evaluateMoment__(1)

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

  def getSensitivities(self):
    """
      Calculates the Sobol indices (percent partial variances) of the terms in this expansion.
      @ In, None
      @ Out, getSensitivities, tuple(dict), Sobol indices and partial variances keyed by subset
    """
    if self.sdx is not None and self.partialVariances is not None:
      self.raiseADebug('Using previously-constructed ANOVA terms...')
      return self.sdx,self.partialVariances
    self.raiseADebug('Constructing ANOVA terms...')
    #collect terms
    terms = {}
    allFalse = tuple(False for _ in self.features)
    for subset,mult in self.reducedTerms.items():
      #skip mean, since it will be subtracted off in the end
      if subset == (): continue
      for poly,coeff in self.ROMs[subset].polyCoeffDict.items():
        #skip mean terms
        if sum(poly) == 0: continue
        poly = self.__fillIndexWithRef(subset,poly)
        polySubset = self._polyToSubset(poly)
        if polySubset not in terms.keys(): terms[polySubset] = {}
        if poly not in terms[polySubset].keys(): terms[polySubset][poly] = 0
        terms[polySubset][poly] += coeff*mult
    #calculate partial variances
    self.partialVariances = {}
    for subset in terms.keys():
      self.partialVariances[subset] = sum(v*v for v in terms[subset].values())
    #calculate indices
    totVar = sum(self.partialVariances.values())
    self.sdx = {}
    for subset,value in self.partialVariances.items():
      self.sdx[subset] = value / totVar
    return self.sdx,self.partialVariances

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
    self.__amsc             = None
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

    names = [name.encode('ascii') for name in self.features + [self.target]]
    # Data is already normalized, so ignore this parameter
    ### Comment replicated from the post-processor version, not sure what it
    ### means (DM)
    # FIXME: AMSC_Object employs unsupervised NearestNeighbors algorithm from
    #        scikit learn.
    #        The NearestNeighbor algorithm is implemented in
    #        SupervisedLearning, which requires features and targets by
    #        default, which we don't have here. When the NearestNeighbor is
    #        implemented in unSupervisedLearning switch to it.
    self.__amsc = AMSC_Object(X=self.X, Y=self.Y, w=weights, names=names,
                              graph=self.graph, gradient=self.gradient,
                              knn=self.knn, beta=self.beta,
                              normalization=None,
                              persistence=self.persistence)
    self.__amsc.Persistence(self.simplification)
    self.__amsc.BuildLinearModels(self.simplification)

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
      @ Out, predictions, numpy.array 1-D, predicted values
    """
    if self.partitionPredictor == 'kde':
      partitions = self.__amsc.Partitions(self.simplification)
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
          fx = self.__amsc.Predict(featureVals,key)
          wx = weights[key]
          sumW += wx
          weightedPredictions += fx*wx
        if sumW == 0:
          return weightedPredictions
        return weightedPredictions / sumW
      else:
        predictions = np.zeros(featureVals.shape[0])
        maxWeights = np.zeros(featureVals.shape[0])
        for key in partitions.keys():
          fx = self.__amsc.Predict(featureVals,key)
          wx = weights[key]
          predictions[wx > maxWeights] = fx
          maxWeights[wx > maxWeights] = wx
        return predictions
    elif self.partitionPredictor == 'svm':
      partitions = self.__amsc.Partitions(self.simplification)
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
          fx = self.__amsc.Predict(featureVals,key)
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

        return weightedPredictions/sumW
      else:
        predictions = np.zeros(featureVals.shape[0])
        maxWeights = np.zeros(featureVals.shape[0])
        for idx,key in enumerate(partitions.keys()):
          fx = self.__amsc.Predict(featureVals,key)
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

        return predictions


  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    self.X      = []
    self.Y      = []
    self.__amsc = None
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
    self.interpolator = interpolationND.NDspline()

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    self.interpolator.reset()
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
    self.interpolator = interpolationND.InverseDistanceWeighting(float(self.initOptionDict['p']))

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    self.interpolator.reset(float(self.initOptionDict['p']))
#
#
#
class SciKitLearn(superVisedLearning):
  """
  An Interface to the ROMs provided by skLearn
  """
  ## The types in this list should not be normalized by default. I would argue
  ## that none of these should be normed by default, since sklearn offers that
  ## option where applicable, but that is for someone else to make a decision.
  unnormedTypes = ['MultinomialNB']

  ROMtype = 'SciKitLearn'
  availImpl = {}
  availImpl['lda'] = {}
  availImpl['lda']['LDA'] = (lda.LDA, 'int') #Quadratic Discriminant Analysis (QDA)
  availImpl['linear_model'] = {} #Generalized Linear Models
  availImpl['linear_model']['ARDRegression'               ] = (linear_model.ARDRegression               , 'float'  ) #Bayesian ARD regression.
  availImpl['linear_model']['BayesianRidge'               ] = (linear_model.BayesianRidge               , 'float'  ) #Bayesian ridge regression
  availImpl['linear_model']['ElasticNet'                  ] = (linear_model.ElasticNet                  , 'float'  ) #Linear Model trained with L1 and L2 prior as regularizer
  availImpl['linear_model']['ElasticNetCV'                ] = (linear_model.ElasticNetCV                , 'float'  ) #Elastic Net model with iterative fitting along a regularization path
  availImpl['linear_model']['Lars'                        ] = (linear_model.Lars                        , 'float'  ) #Least Angle Regression model a.k.a.
  availImpl['linear_model']['LarsCV'                      ] = (linear_model.LarsCV                      , 'float'  ) #Cross-validated Least Angle Regression model
  availImpl['linear_model']['Lasso'                       ] = (linear_model.Lasso                       , 'float'  ) #Linear Model trained with L1 prior as regularizer (aka the Lasso)
  availImpl['linear_model']['LassoCV'                     ] = (linear_model.LassoCV                     , 'float'  ) #Lasso linear model with iterative fitting along a regularization path
  availImpl['linear_model']['LassoLars'                   ] = (linear_model.LassoLars                   , 'float'  ) #Lasso model fit with Least Angle Regression a.k.a.
  availImpl['linear_model']['LassoLarsCV'                 ] = (linear_model.LassoLarsCV                 , 'float'  ) #Cross-validated Lasso, using the LARS algorithm
  availImpl['linear_model']['LassoLarsIC'                 ] = (linear_model.LassoLarsIC                 , 'float'  ) #Lasso model fit with Lars using BIC or AIC for model selection
  availImpl['linear_model']['LinearRegression'            ] = (linear_model.LinearRegression            , 'float'  ) #Ordinary least squares Linear Regression.
  availImpl['linear_model']['LogisticRegression'          ] = (linear_model.LogisticRegression          , 'float'  ) #Logistic Regression (aka logit, MaxEnt) classifier.
  availImpl['linear_model']['MultiTaskLasso'              ] = (linear_model.MultiTaskLasso              , 'float'  ) #Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer
  availImpl['linear_model']['MultiTaskElasticNet'         ] = (linear_model.MultiTaskElasticNet         , 'float'  ) #Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer
  availImpl['linear_model']['OrthogonalMatchingPursuit'   ] = (linear_model.OrthogonalMatchingPursuit   , 'float'  ) #Orthogonal Mathching Pursuit model (OMP)
  availImpl['linear_model']['OrthogonalMatchingPursuitCV' ] = (linear_model.OrthogonalMatchingPursuitCV , 'float'  ) #Cross-validated Orthogonal Mathching Pursuit model (OMP)
  availImpl['linear_model']['PassiveAggressiveClassifier' ] = (linear_model.PassiveAggressiveClassifier , 'int') #Passive Aggressive Classifier
  availImpl['linear_model']['PassiveAggressiveRegressor'  ] = (linear_model.PassiveAggressiveRegressor  , 'float'  ) #Passive Aggressive Regressor
  availImpl['linear_model']['Perceptron'                  ] = (linear_model.Perceptron                  , 'float'  ) #Perceptron
  availImpl['linear_model']['Ridge'                       ] = (linear_model.Ridge                       , 'float'  ) #Linear least squares with l2 regularization.
  availImpl['linear_model']['RidgeClassifier'             ] = (linear_model.RidgeClassifier             , 'float'  ) #Classifier using Ridge regression.
  availImpl['linear_model']['RidgeClassifierCV'           ] = (linear_model.RidgeClassifierCV           , 'int') #Ridge classifier with built-in cross-validation.
  availImpl['linear_model']['RidgeCV'                     ] = (linear_model.RidgeCV                     , 'float'  ) #Ridge regression with built-in cross-validation.
  availImpl['linear_model']['SGDClassifier'               ] = (linear_model.SGDClassifier               , 'int') #Linear classifiers (SVM, logistic regression, a.o.) with SGD training.
  availImpl['linear_model']['SGDRegressor'                ] = (linear_model.SGDRegressor                , 'float'  ) #Linear model fitted by minimizing a regularized empirical loss with SGD

  availImpl['svm'] = {} #support Vector Machines
  availImpl['svm']['LinearSVC'] = (svm.LinearSVC, 'bool')
  availImpl['svm']['SVC'      ] = (svm.SVC      , 'bool')
  availImpl['svm']['NuSVC'    ] = (svm.NuSVC    , 'bool')
  availImpl['svm']['SVR'      ] = (svm.SVR      , 'bool')

  availImpl['multiClass'] = {} #Multiclass and multilabel classification
  availImpl['multiClass']['OneVsRestClassifier' ] = (multiclass.OneVsRestClassifier , 'int') # One-vs-the-rest (OvR) multiclass/multilabel strategy
  availImpl['multiClass']['OneVsOneClassifier'  ] = (multiclass.OneVsOneClassifier  , 'int') # One-vs-one multiclass strategy
  availImpl['multiClass']['OutputCodeClassifier'] = (multiclass.OutputCodeClassifier, 'int') # (Error-Correcting) Output-Code multiclass strategy
  availImpl['multiClass']['fit_ovr'             ] = (multiclass.fit_ovr             , 'int') # Fit a one-vs-the-rest strategy.
  availImpl['multiClass']['predict_ovr'         ] = (multiclass.predict_ovr         , 'int') # Make predictions using the one-vs-the-rest strategy.
  availImpl['multiClass']['fit_ovo'             ] = (multiclass.fit_ovo             , 'int') # Fit a one-vs-one strategy.
  availImpl['multiClass']['predict_ovo'         ] = (multiclass.predict_ovo         , 'int') # Make predictions using the one-vs-one strategy.
  availImpl['multiClass']['fit_ecoc'            ] = (multiclass.fit_ecoc            , 'int') # Fit an error-correcting output-code strategy.
  availImpl['multiClass']['predict_ecoc'        ] = (multiclass.predict_ecoc        , 'int') # Make predictions using the error-correcting output-code strategy.

  availImpl['naiveBayes'] = {}
  availImpl['naiveBayes']['GaussianNB'   ] = (naive_bayes.GaussianNB   , 'float')
  availImpl['naiveBayes']['MultinomialNB'] = (naive_bayes.MultinomialNB, 'float')
  availImpl['naiveBayes']['BernoulliNB'  ] = (naive_bayes.BernoulliNB  , 'float')

  availImpl['neighbors'] = {}
  availImpl['neighbors']['KNeighborsClassifier']     = (neighbors.KNeighborsClassifier     , 'int')# Classifier implementing the k-nearest neighbors vote.
  availImpl['neighbors']['RadiusNeighbors']          = (neighbors.RadiusNeighborsClassifier, 'int')# Classifier implementing a vote among neighbors within a given radius
  availImpl['neighbors']['KNeighborsRegressor']      = (neighbors.KNeighborsRegressor      , 'float'  )# Regression based on k-nearest neighbors.
  availImpl['neighbors']['RadiusNeighborsRegressor'] = (neighbors.RadiusNeighborsRegressor , 'float'  )# Regression based on neighbors within a fixed radius.
  availImpl['neighbors']['NearestCentroid']          = (neighbors.NearestCentroid          , 'int')# Nearest centroid classifier.
  availImpl['neighbors']['BallTree']                 = (neighbors.BallTree                 , 'float'  )# BallTree for fast generalized N-point problems
  availImpl['neighbors']['KDTree']                   = (neighbors.KDTree                   , 'float'  )# KDTree for fast generalized N-point problems

  availImpl['qda'] = {}
  availImpl['qda']['QDA'] = (qda.QDA, 'int') #Quadratic Discriminant Analysis (QDA)

  availImpl['tree'] = {}
  availImpl['tree']['DecisionTreeClassifier'] = (tree.DecisionTreeClassifier, 'int')# A decision tree classifier.
  availImpl['tree']['DecisionTreeRegressor' ] = (tree.DecisionTreeRegressor , 'float'  )# A tree regressor.
  availImpl['tree']['ExtraTreeClassifier'   ] = (tree.ExtraTreeClassifier   , 'int')# An extremely randomized tree classifier.
  availImpl['tree']['ExtraTreeRegressor'    ] = (tree.ExtraTreeRegressor    , 'float'  )# An extremely randomized tree regressor.

  availImpl['GaussianProcess'] = {}
  availImpl['GaussianProcess']['GaussianProcess'] = (gaussian_process.GaussianProcess    , 'float'  )
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
      A constructor that will appropriately intialize a supervised learning object
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
    self.initOptionDict.pop('SKLtype')
    if not SKLtype in self.__class__.availImpl.keys():
      self.raiseAnError(IOError,'not known SKLtype "' + SKLtype +'" (from ROM "'+name+'")')
    if not SKLsubType in self.__class__.availImpl[SKLtype].keys():
      self.raiseAnError(IOError,'not known SKLsubType "' + SKLsubType +'" (from ROM "'+name+'")')

    self.__class__.returnType     = self.__class__.availImpl[SKLtype][SKLsubType][1]
    self.__class__.qualityEstType = self.__class__.qualityEstTypeDict[SKLtype][SKLsubType]

    if 'estimator' in self.initOptionDict.keys():
      estimatorDict = self.initOptionDict['estimator']
      self.initOptionDict.pop('estimator')
      estimatorSKLtype, estimatorSKLsubType = estimatorDict['SKLtype'].split('|')
      estimator = self.__class__.availImpl[estimatorSKLtype][estimatorSKLsubType][0]()
      self.ROM = self.__class__.availImpl[SKLtype][SKLsubType][0](estimator)
    else:
      self.ROM  = self.__class__.availImpl[SKLtype][SKLsubType][0]()

    for key,value in self.initOptionDict.items():
      try:self.initOptionDict[key] = ast.literal_eval(value)
      except: pass
    self.ROM.set_params(**self.initOptionDict)

  def _readdressEvaluateConstResponse(self,edict):
    """
      Method to re-address the evaluate base class method in order to avoid wasting time
      in case the training set has an unique response (e.g. if 10 points in the training set,
      and the 10 outcomes are all == to 1, this method returns one without the need of an
      evaluation)
      @ In, edict, dict, prediction request. Not used in this method (kept the consistency with evaluate method)
      @ Out, myNumber, float, the evaluation
    """
    return self.myNumber

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
      @ Out, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """
    #If all the target values are the same no training is needed and the moreover the self.evaluate could be re-addressed to this value
    if len(np.unique(targetVals))>1:
      self.ROM.fit(featureVals,targetVals)
      self.evaluate = self._readdressEvaluateRomResponse
      #self.evaluate = lambda edict : self.__class__.evaluate(self,edict)
    else:
      self.myNumber = np.unique(targetVals)[0]
      self.evaluate = self._readdressEvaluateConstResponse

  def __confidenceLocal__(self,featureVals):
    """
      This should return an estimation of the quality of the prediction.
      @ In, featureVals, 2-D numpy array, [n_samples,n_features]
      @ Out, predict_proba, float, the confidence
    """
    if  'probability' in self.__class__.qualityEstType: return self.ROM.predict_proba(featureVals)
    else            : self.raiseAnError(IOError,'the ROM '+str(self.initOptionDict['name'])+'has not the an method to evaluate the confidence of the prediction')

  def __evaluateLocal__(self,featureVals):
    """
      Evaluates a point.
      @ In, featureVals, np.array, list of values at which to evaluate the ROM
      @ Out, predict, float, the evaluated value
    """
    return self.ROM.predict(featureVals)

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    self.ROM = self.ROM.__class__(**self.initOptionDict)

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    params = self.ROM.get_params()
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
    if self.subType in self.unnormedTypes:
      self.muAndSigmaFeatures[feat] = (0.0,1.0)
    else:
      super(SciKitLearn, self)._localNormalizeData(values,names,feat)
#
#
#
__interfaceDict                         = {}
__interfaceDict['NDspline'            ] = NDsplineRom
__interfaceDict['NDinvDistWeight'     ] = NDinvDistWeight
__interfaceDict['SciKitLearn'         ] = SciKitLearn
__interfaceDict['GaussPolynomialRom'  ] = GaussPolynomialRom
__interfaceDict['HDMRRom'             ] = HDMRRom
__interfaceDict['MSR'                 ] = MSR
__base                                  = 'superVisedLearning'

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
