"""
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
import sys
import numpy as np
import numpy
import abc
import ast
import pickle as pk
from operator import itemgetter
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import metaclass_insert, returnPrintTag, returnPrintPostTag, find_interpolationND,stringsThatMeanFalse
interpolationND = find_interpolationND()
#Internal Modules End--------------------------------------------------------------------------------

class superVisedLearning(metaclass_insert(abc.ABCMeta)):
  """
  This is the general interface to any superVisedLearning learning method.
  Essentially it contains a train, and evaluate methods
  """
  returnType      = '' #this describe the type of information generated the possibility are 'boolean', 'integer', 'float'
  qualityEstType  = [] #this describe the type of estimator returned known type are 'distance', 'probability'. The values are returned by the self.__confidenceLocal__(Features)
  ROMtype         = '' #the broad class of the interpolator

  @staticmethod
  def checkArrayConsistency(arrayin):
    """
    This method checks the consistency of the in-array
    @ In, object... It should be an array
    @ Out, tuple, tuple[0] is a bool (True -> everything is ok, False -> something wrong), tuple[1], string ,the error mesg
    """
    if type(arrayin) != numpy.ndarray: return (False,' The object is not a numpy array')
    if len(arrayin.shape) > 1: return(False, ' The array must be 1-d')
    return (True,'')

  def __init__(self,**kwargs):
    self.printTag = returnPrintTag('SuperVised')
    #booleanFlag that controls the normalization procedure. If true, the normalization is performed. Default = True
    if kwargs != None: self.initOptionDict = kwargs
    else             : self.initOptionDict = {}
    if 'Features' not in self.initOptionDict.keys(): raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> Feature names not provided')
    if 'Target'   not in self.initOptionDict.keys(): raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> Target name not provided')
    self.features = self.initOptionDict['Features'].split(',')
    self.target   = self.initOptionDict['Target'  ]
    self.initOptionDict.pop('Target')
    self.initOptionDict.pop('Features')
    if self.features.count(self.target) > 0: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> The target and one of the features have the same name!!!!')
    #average value and sigma are used for normalization of the feature data
    #a dictionary where for each feature a tuple (average value, sigma)
    self.muAndSigmaFeatures = {}
    #these need to be declared in the child classes!!!!
    self.amITrained         = False

  def initialize(self,idict):
    pass #Overloaded by (at least) GaussPolynomialRom

  def train(self,tdict):
    """
      Method to perform the training of the superVisedLearning algorithm
      NB.the superVisedLearning object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires. So far the base class will do the translation into numpy
      @ In, tdict, training dictionary
      @ Out, None
    """
    if type(tdict) != dict: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> method "train". The training set needs to be provided through a dictionary. Type of the in-object is ' + str(type(tdict)))
    names, values  = list(tdict.keys()), list(tdict.values())
    if self.target in names: targetValues = values[names.index(self.target)]
    else                   : raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> The output sought '+self.target+' is not in the training set')
    # check if the targetValues are consistent with the expected structure
    resp = self.checkArrayConsistency(targetValues)
    if not resp[0]: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> In training set for target '+self.target+':'+resp[1])
    # construct the evaluation matrixes
    featureValues = np.zeros(shape=(targetValues.size,len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> The feature sought '+feat+' is not in the training set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> In training set for feature '+feat+':'+resp[1])
        if values[names.index(feat)].size != featureValues[:,0].size: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> In training set, the number of values provided for feature '+feat+' are != number of target outcomes!')
        self._localNormalizeData(values,names,feat)
        if self.muAndSigmaFeatures[feat][1]==0: self.muAndSigmaFeatures[feat] = (self.muAndSigmaFeatures[feat][0],np.max(np.absolute(values[names.index(feat)])))
        if self.muAndSigmaFeatures[feat][1]==0: self.muAndSigmaFeatures[feat] = (self.muAndSigmaFeatures[feat][0],1.0)
        featureValues[:,cnt] = (values[names.index(feat)] - self.muAndSigmaFeatures[feat][0])/self.muAndSigmaFeatures[feat][1]
    self.__trainLocal__(featureValues,targetValues)
    self.amITrained = True

  def _localNormalizeData(self,values,names,feat):
    """
    Method to normalize data based on the mean and standard deviation.  If undesired for a particular ROM,
    this method can be overloaded to simply pass (see, e.g., GaussPolynomialRom).
    @ In, values, list of feature values (from tdict)
    @ In, names, names of features (from tdict)
    @ In, feat, list of features (from ROM)
    @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (np.average(values[names.index(feat)]),np.std(values[names.index(feat)]))

  def confidence(self,edict):
    """
    This call is used to get an estimate of the confidence in the prediction.
    The base class self.confidence will translate a dictionary into numpy array, then call the local confidence
    """
    if type(edict) != dict: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> method "confidence". The inquiring set needs to be provided through a dictionary. Type of the in-object is ' + str(type(edict)))
    names, values   = list(edict.keys()), list(edict.values())
    for index in range(len(values)):
      resp = self.checkArrayConsistency(values[index])
      if not resp[0]: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> In evaluate request for feature '+names[index]+':'+resp[1])
    featureValues = np.zeros(shape=(values[0].size,len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> The feature sought '+feat+' is not in the evaluate set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> In training set for feature '+feat+':'+resp[1])
        featureValues[:,cnt] = values[names.index(feat)]
    return self.__confidenceLocal__(featureValues)

  def evaluate(self,edict):
    """
    Method to perform the evaluation of a point or a set of points through the previous trained superVisedLearning algorithm
    NB.the superVisedLearning object is committed to convert the dictionary that is passed (in), into the local format
    the interface with the kernels requires.
    @ In, tdict, evaluation dictionary
    @ Out, numpy array of evaluated points
    """
    if type(edict) != dict: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> method "evaluate". The evaluate request/s need/s to be provided through a dictionary. Type of the in-object is ' + str(type(edict)))
    names, values  = list(edict.keys()), list(edict.values())
    for index in range(len(values)):
      resp = self.checkArrayConsistency(values[index])
      if not resp[0]: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> In evaluate request for feature '+names[index]+':'+resp[1])
    # construct the evaluation matrix
    featureValues = np.zeros(shape=(values[0].size,len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> The feature sought '+feat+' is not in the evaluate set')
      else:
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]: raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> In training set for feature '+feat+':'+resp[1])
        featureValues[:,cnt] = ((values[names.index(feat)] - self.muAndSigmaFeatures[feat][0]))/self.muAndSigmaFeatures[feat][1]
    return self.__evaluateLocal__(featureValues)

  def reset(self):
    """override this method to re-instance the ROM"""
    self.amITrained = False
    self.__resetLocal__()

  def returnInitialParameters(self):
    """override this method to return the fix set of parameters of the ROM"""
    iniParDict = dict(self.initOptionDict.items() + {'returnType':self.__class__.returnType,'qualityEstType':self.__class__.qualityEstType,'Features':self.features,
                                             'Target':self.target,'returnType':self.__class__.returnType}.items() + self.__returnInitialParametersLocal__().items())
    return iniParDict

  def returnCurrentSetting(self):
    """return the set of parameters of the ROM that can change during simulation"""
    return dict({'Trained':self.amITrained}.items() + self.__CurrentSettingDictLocal__().items())

  @abc.abstractmethod
  def __trainLocal__(self,featureVals,targetVals):
    """@ In, featureVals, 2-D numpy array [n_samples,n_features]"""

  @abc.abstractmethod
  def __confidenceLocal__(self,featureVals):
    """
    This should return an estimation of the quality of the prediction.
    This could be distance or probability or anything else, the type needs to be declared in the variable cls.qualityEstType
    @ In, featureVals, 2-D numpy array [n_samples,n_features]
    """

  @abc.abstractmethod
  def __evaluateLocal__(self,featureVals):
    """
    @ In,  featureVals, 2-D numpy array [n_samples,n_features]
    @ Out, targetVals , 1-D numpy array [n_samples]
    """

  @abc.abstractmethod
  def __resetLocal__(self,featureVals):
    """After this method the ROM should be described only by the initial parameter settings"""

  @abc.abstractmethod
  def __returnInitialParametersLocal__(self):
    """this should return a dictionary with the parameters that could be possible not in self.initOptionDict"""

  @abc.abstractmethod
  def __returnCurrentSettingLocal__(self):
    """override this method to pass the set of parameters of the ROM that can change during simulation"""
#
#
#
class NDinterpolatorRom(superVisedLearning):
  def __init__(self,**kwargs):
    superVisedLearning.__init__(self,**kwargs)
    self.interpolator = None
    self.printTag = returnPrintTag('ND Interpolation ROM')

  def __trainLocal__(self,featureVals,targetVals):
    """
    Perform training on samples in X with responses y.
    For an one-class model, +1 or -1 is returned.
    Parameters: featureVals : {array-like, sparse matrix}, shape = [n_samples, n_features]
    Returns   : targetVals : array, shape = [n_samples]
    """
    featv = interpolationND.vectd2d(featureVals[:][:])
    targv = interpolationND.vectd(targetVals)
    self.interpolator.fit(featv,targv)

  def __confidenceLocal__(self,featureVals):
    raise NotImplemented('NDinterpRom   : __confidenceLocal__ method must be implemented!!!!!')

  def __evaluateLocal__(self,featureVals):
    """
    Perform regression on samples in featureVals.
    For an one-class model, +1 or -1 is returned.
    @ In, numpy.array 2-D, features
    @ Out, numpy.array 1-D, predicted values
    """
    prediction = np.zeros(featureVals.shape[0])
    for n_sample in range(featureVals.shape[0]):
      featv = interpolationND.vectd(featureVals[n_sample][:])
      prediction[n_sample] = self.interpolator.interpolateAt(featv)
      print('NDinterpRom   : Prediction by ' + self.__class__.ROMtype + '. Predicted value is ' + str(prediction[n_sample]))
    return prediction

  def __returnInitialParametersLocal__(self):
    """there are no possible default parameters to report"""
    localInitParam = {}
    return localInitParam

  def __returnCurrentSettingLocal__(self):
    raise NotImplemented('NDinterpRom   : __returnCurrentSettingLocal__ method must be implemented!!!!!')

class GaussPolynomialRom(NDinterpolatorRom):
  def __confidenceLocal__(self,edict):pass #TODO

  def __resetLocal__(self):pass #TODO

  def __returnCurrentSettingLocal__(self):pass #TODO

  def __init__(self,**kwargs):
    superVisedLearning.__init__(self,**kwargs)
    self.interpolator  = None #FIXME what's this?
    self.printTag      = returnPrintTag('GAUSSgpcROM('+self.target+')')
    self.indexSetType  = None #string of index set type, TensorProduct or TotalDegree or HyperbolicCross
    self.maxPolyOrder  = None #integer of relative maximum polynomial order to use in any one dimension
    self.itpDict       = {}   #dict of quad,poly,weight choices keyed on varName
    self.norm          = None #combined distribution normalization factors (product)
    self.sparseGrid    = None #Quadratures.SparseGrid object, has points and weights
    self.distDict      = None #dict{varName: Distribution object}, has point conversion methods based on quadrature
    self.quads         = None #dict{varName: Quadrature object}, has keys for distribution's point conversion methods
    self.polys         = None #dict{varName: OrthoPolynomial object}, has polynomials for evaluation
    self.indexSet      = None #array of tuples, polynomial order combinations
    self.polyCoeffDict = None #dict{index set point, float}, polynomial combination coefficients for each combination
    self.itpDict       = {}   #dict{varName: dict{attribName:value} }

    for key,val in kwargs.items():
      if key=='IndexSet': self.indexSetType = val
      if key=='PolynomialOrder': self.maxPolyOrder = val
      if key=='Interpolation':
        for var,val in val.items():
          self.itpDict[var]={'poly'  :'DEFAULT',
                             'quad'  :'DEFAULT',
                             'weight':'1'}
                             #'cdf'   :'False'}
          for atrName,atrVal in val.items():
            if atrName in ['poly','quad','weight']: self.itpDict[var][atrName]=atrVal
            else: raise IOError(self.printTag+' Unrecognized option: '+atrName)

    if not self.indexSetType:
      raise IOError(self.printTag+' No IndexSet specified!')
    if not self.maxPolyOrder:
      raise IOError(self.printTag+' No maxPolyOrder specified!')
    if self.maxPolyOrder < 1:
      raise IOError(self.printTag+' Polynomial order cannot be less than 1 currently.')

  def _localNormalizeData(self,values,names,feat):
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def interpolationInfo(self):
    return dict(self.itpDict)

  def initialize(self,idict):
    for key,value in idict.items():
      if   key == 'SG'   : self.sparseGrid = value
      elif key == 'dists': self.distDict   = value
      elif key == 'quads': self.quads      = value
      elif key == 'polys': self.polys      = value
      elif key == 'iSet' : self.indexSet   = value

  def _multiDPolyBasisEval(self,orders,pts):
    """Evaluates each polynomial set at given orders and points, returns product.
    @ In orders, tuple(int), polynomial orders to evaluate
    @ In pts, tuple(float), values at which to evaluate polynomials
    @ Out, float, product of polynomial evaluations
    """
    tot=1
    for i,(o,p) in enumerate(zip(orders,pts)):
      varName = self.sparseGrid.varNames[i]
      tot*=self.polys[varName](o,p)
    return tot

  def __trainLocal__(self,featureVals,targetVals):
    self.polyCoeffDict={}
    #check consistency of featureVals
    if len(featureVals)!=len(self.sparseGrid):
      raise IOError(self.printTag+' ERROR: ROM requires '+str(len(self.sparseGrid))+' points, but only '+str(len(featureVals))+' provided!')
    #the dimensions of featureVals might be reordered from sparseGrid, so fix it here
    self.sparseGrid._remap(self.features)
    #check equality of point space
    fvs = featureVals[:]
    sgs = self.sparseGrid.points()[:]
    fvs=sorted(fvs,key=itemgetter(*range(len(fvs[0]))))
    sgs=sorted(sgs,key=itemgetter(*range(len(sgs[0]))))
    #print('DEBUG fvs | sgs:')
    #for i in range(len(fvs)):
    #  print('  ',fvs[i],' | ',sgs[i])
    if not np.allclose(fvs,sgs,rtol=1e-15):
      print('DEBUG fvs | sgs:')
      for i in range(len(fvs)):
        print('  ',fvs[i],' | ',sgs[i])
      raise IOError(self.printTag+' ERROR: input values do not match required values!')
    #make translation matrix between lists
    translate={}
    for i in range(len(fvs)):
      translate[tuple(fvs[i])]=sgs[i]
    #TODO can parallelize this! Worth it?
    self.norm = np.prod(list(self.distDict[v].measureNorm(self.quads[v].type) for v in self.distDict.keys()))
    #outFile=file('debugout.txt','w')
    #outFile.writelines(str(list(v for v in self.sparseGrid.varNames))+'\n')
    for i,idx in enumerate(self.indexSet):
      idx=tuple(idx)
      self.polyCoeffDict[idx]=0
      wtsum=0
      for pt,soln in zip(featureVals,targetVals):
        stdPt = np.zeros(len(pt))
        for i,p in enumerate(pt):
          varName = self.sparseGrid.varNames[i]
          stdPt[i] = self.distDict[varName].convertToQuad(self.quads[varName].type,p)
        #outFile.writelines('DEBUG pt,stdpt\n')
        #outFile.writelines('  '+str(pt)+'\n')
        #outFile.writelines('  '+str(stdPt)+'\n')
        wt = self.sparseGrid.weights(translate[tuple(pt)])
        self.polyCoeffDict[idx]+=soln*self._multiDPolyBasisEval(idx,stdPt)*wt
      self.polyCoeffDict[idx]*=self.norm
    #outFile.close()
    self.printPolyDict()
    #do a few moments #TODO need a better solution for calling moment calculations, etc
    for r in range(5):
      print('ROM moment',r,'= %1.16f' %self.__evaluateMoment__(r))

    #print('DEBUG sparsegrid:\n'+str(self.sparseGrid))
    #print('DEBUG scipy tests')
    #for i in range(1,5):
    #  print('DEBUG  ',self.quads.values()[0](i))
    #import scipy
    #print('DEBUG python',sys.version_info)
    #print('DEBUG numpy ',np.__version__)
    #print('DEBUG scipy ',scipy.__version__)
    #local evals
    #if len(self.features)==1:
    #  tests=[(0),(0.2),(0.5),(0.7),(1.0)]
    #elif len(self.features)==2:
    # tests=[(0,0),(0,0.5),(0.5,0.5),(1.0,0.5),(1.0,1.0)]
    #for i in tests:
    #  print('DEBUG eval'+str(i)+':',self.__evaluateLocal__([i]))

    #pk.dump(self,file('testROMdump.pk','w'))
    #dud=pk.load(file('testROMdump.pk','r'))
    #print('dud:',dud)
    #print('pickle test:',dud==self.distDict)

    #for r in range(5):
    #  print('ROM moment',r,'= %1.16f' %dud.__evaluateMoment__(r))
    # THESE members pickle ok now, and check out as equal
    # - sparseGrid
    # - quads
    # - maxPolyOrder
    # - polys
    # - indexSet

    # THESE members are problems still. TODO
    # - distDict -> SWIG problems.

    # UNTESTED
    # SVL

  def printPolyDict(self,printZeros=False):
    """Human-readable version of the polynomial chaos expansion.
    @ In printZeros,boolean,optional flag for printing even zero coefficients
    @ Out, None, None
    """
    data=[]
    for idx,val in self.polyCoeffDict.items():
      if val > 1e-14 or printZeros:
        data.append([idx,val])
    data.sort()
    print('polyDict for '+self.target+': ')
    for idx,val in data:
      print('    ',idx,val)

  def __evaluateMoment__(self,r):
    """Use the ROM's built-in method to calculate moments.
    @ In r, int, moment to calculate
    @ Out, float, evaluation of moment
    """
    #TODO is there a faster way still to do this?
    tot=0
    for pt,wt in self.sparseGrid:
      tot+=self.__evaluateLocal__([pt])**r*wt
    tot*=self.norm
    return tot

  def __evaluateLocal__(self,featureVals):
    featureVals=featureVals[0]
    tot=0
    stdPt = np.zeros(len(featureVals))
    for p,pt in enumerate(featureVals):
      varName = self.sparseGrid.varNames[p]
      stdPt[p] = self.distDict[varName].convertToQuad(self.quads[varName].type,pt)
    for idx,coeff in self.polyCoeffDict.items():
      tot+=coeff*self._multiDPolyBasisEval(idx,stdPt)
    return tot

  def __returnInitialParametersLocal__(self):
    return {}#TODO 'IndexSet:':self.indexSetType,
             #'PolynomialOrder':self.maxPolyOrder,
             # 'Interpolation':interpolationInfo()}

  #def __getstate__(self): #for pickling
  #  pass
#
#  def __setstate__(self,state): #for unpickling
#    pass
#
#
#
class NDsplineRom(NDinterpolatorRom):
  ROMtype         = 'NDsplineRom'
  def __init__(self,**kwargs):
    NDinterpolatorRom.__init__(self,**kwargs)
    self.printTag = returnPrintTag('ND-SPLINE ROM')
    self.interpolator = interpolationND.NDspline()

  def __resetLocal__(self):
    """ The reset here erase the Interpolator while keeping the instance"""
    self.interpolator.reset()
#
#
#
class NDinvDistWeigth(NDinterpolatorRom):
  ROMtype         = 'NDinvDistWeigth'
  def __init__(self,**kwargs):
    NDinterpolatorRom.__init__(self,**kwargs)
    self.printTag = returnPrintTag('ND-INVERSEWEIGHT ROM')
    if not 'p' in self.initOptionDict.keys(): raise IOError (self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> the <p> parameter must be provided in order to use NDinvDistWeigth as ROM!!!!')
    self.interpolator = interpolationND.InverseDistanceWeighting(float(self.initOptionDict['p']))

  def __resetLocal__(self):
    """ The reset here erase the Interpolator while keeping the instance"""
    self.interpolator.reset(float(self.initOptionDict['p']))
#
#
#
class NDmicroSphere(NDinterpolatorRom):
  ROMtype         = 'NDmicroSphere'
  def __init__(self,**kwargs):
    NDinterpolatorRom.__init__(self,**kwargs)
    self.printTag = returnPrintTag('ND-MICROSPHERE ROM')
    if not 'p' in self.initOptionDict.keys(): raise IOError (self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> the <p> parameter must be provided in order to use NDmicroSphere as ROM!!!!')
    if not 'precision' in self.initOptionDict.keys(): raise IOError (self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> the <precision> parameter must be provided in order to use NDmicroSphere as ROM!!!!')
    self.interpolator = interpolationND.microSphere(float(self.initOptionDict['p']),int(self.initOptionDict['precision']))

  def __resetLocal__(self):
    self.interpolator.reset(float(self.initOptionDict['p']),int(self.initOptionDict['precision']))

class SciKitLearn(superVisedLearning):
  ROMtype = 'SciKitLearn'
  availImpl = {}
  availImpl['lda'] = {}
  availImpl['lda']['LDA'] = (lda.LDA, 'integer') #Quadratic Discriminant Analysis (QDA)
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
  availImpl['linear_model']['PassiveAggressiveClassifier' ] = (linear_model.PassiveAggressiveClassifier , 'integer') #Passive Aggressive Classifier
  availImpl['linear_model']['PassiveAggressiveRegressor'  ] = (linear_model.PassiveAggressiveRegressor  , 'float'  ) #Passive Aggressive Regressor
  availImpl['linear_model']['Perceptron'                  ] = (linear_model.Perceptron                  , 'float'  ) #Perceptron
  availImpl['linear_model']['RandomizedLasso'             ] = (linear_model.RandomizedLasso             , 'float'  ) #Randomized Lasso.
  availImpl['linear_model']['RandomizedLogisticRegression'] = (linear_model.RandomizedLogisticRegression, 'float'  ) #Randomized Logistic Regression
  availImpl['linear_model']['Ridge'                       ] = (linear_model.Ridge                       , 'float'  ) #Linear least squares with l2 regularization.
  availImpl['linear_model']['RidgeClassifier'             ] = (linear_model.RidgeClassifier             , 'float'  ) #Classifier using Ridge regression.
  availImpl['linear_model']['RidgeClassifierCV'           ] = (linear_model.RidgeClassifierCV           , 'integer') #Ridge classifier with built-in cross-validation.
  availImpl['linear_model']['RidgeCV'                     ] = (linear_model.RidgeCV                     , 'float'  ) #Ridge regression with built-in cross-validation.
  availImpl['linear_model']['SGDClassifier'               ] = (linear_model.SGDClassifier               , 'integer') #Linear classifiers (SVM, logistic regression, a.o.) with SGD training.
  availImpl['linear_model']['SGDRegressor'                ] = (linear_model.SGDRegressor                , 'float'  ) #Linear model fitted by minimizing a regularized empirical loss with SGD
  availImpl['linear_model']['lars_path'                   ] = (linear_model.lars_path                   , 'float'  ) #Compute Least Angle Regression or Lasso path using LARS algorithm [1]
  availImpl['linear_model']['lasso_path'                  ] = (linear_model.lasso_path                  , 'float'  ) #Compute Lasso path with coordinate descent
  availImpl['linear_model']['lasso_stability_path'        ] = (linear_model.lasso_stability_path        , 'float'  ) #Stabiliy path based on randomized Lasso estimates
  availImpl['linear_model']['orthogonal_mp_gram'          ] = (linear_model.orthogonal_mp_gram          , 'float'  ) #Gram Orthogonal Matching Pursuit (OMP)

  availImpl['svm'] = {} #support Vector Machines
  availImpl['svm']['LinearSVC'] = (svm.LinearSVC, 'boolean')
  availImpl['svm']['SVC'      ] = (svm.SVC      , 'boolean')
  availImpl['svm']['NuSVC'    ] = (svm.NuSVC    , 'boolean')
  availImpl['svm']['SVR'      ] = (svm.SVR      , 'boolean')

  availImpl['multiClass'] = {} #Multiclass and multilabel classification
  availImpl['multiClass']['OneVsRestClassifier' ] = (multiclass.OneVsRestClassifier , 'integer') # One-vs-the-rest (OvR) multiclass/multilabel strategy
  availImpl['multiClass']['OneVsOneClassifier'  ] = (multiclass.OneVsOneClassifier  , 'integer') # One-vs-one multiclass strategy
  availImpl['multiClass']['OutputCodeClassifier'] = (multiclass.OutputCodeClassifier, 'integer') # (Error-Correcting) Output-Code multiclass strategy
  availImpl['multiClass']['fit_ovr'             ] = (multiclass.fit_ovr             , 'integer') # Fit a one-vs-the-rest strategy.
  availImpl['multiClass']['predict_ovr'         ] = (multiclass.predict_ovr         , 'integer') # Make predictions using the one-vs-the-rest strategy.
  availImpl['multiClass']['fit_ovo'             ] = (multiclass.fit_ovo             , 'integer') # Fit a one-vs-one strategy.
  availImpl['multiClass']['predict_ovo'         ] = (multiclass.predict_ovo         , 'integer') # Make predictions using the one-vs-one strategy.
  availImpl['multiClass']['fit_ecoc'            ] = (multiclass.fit_ecoc            , 'integer') # Fit an error-correcting output-code strategy.
  availImpl['multiClass']['predict_ecoc'        ] = (multiclass.predict_ecoc        , 'integer') # Make predictions using the error-correcting output-code strategy.

  availImpl['naiveBayes'] = {}
  availImpl['naiveBayes']['GaussianNB'   ] = (naive_bayes.GaussianNB   , 'float')
  availImpl['naiveBayes']['MultinomialNB'] = (naive_bayes.MultinomialNB, 'float')
  availImpl['naiveBayes']['BernoulliNB'  ] = (naive_bayes.BernoulliNB  , 'float')

  availImpl['neighbors'] = {}
  availImpl['neighbors']['NearestNeighbors']         = (neighbors.NearestNeighbors         , 'float'  )# Unsupervised learner for implementing neighbor searches.
  availImpl['neighbors']['KNeighborsClassifier']     = (neighbors.KNeighborsClassifier     , 'integer')# Classifier implementing the k-nearest neighbors vote.
  availImpl['neighbors']['RadiusNeighbors']          = (neighbors.RadiusNeighborsClassifier, 'integer')# Classifier implementing a vote among neighbors within a given radius
  availImpl['neighbors']['KNeighborsRegressor']      = (neighbors.KNeighborsRegressor      , 'float'  )# Regression based on k-nearest neighbors.
  availImpl['neighbors']['RadiusNeighborsRegressor'] = (neighbors.RadiusNeighborsRegressor , 'float'  )# Regression based on neighbors within a fixed radius.
  availImpl['neighbors']['NearestCentroid']          = (neighbors.NearestCentroid          , 'integer')# Nearest centroid classifier.
  availImpl['neighbors']['BallTree']                 = (neighbors.BallTree                 , 'float'  )# BallTree for fast generalized N-point problems
  availImpl['neighbors']['KDTree']                   = (neighbors.KDTree                   , 'float'  )# KDTree for fast generalized N-point problems

  availImpl['qda'] = {}
  availImpl['qda']['QDA'] = (qda.QDA, 'integer') #Quadratic Discriminant Analysis (QDA)

  availImpl['tree'] = {}
  availImpl['tree']['DecisionTreeClassifier'] = (tree.DecisionTreeClassifier, 'integer')# A decision tree classifier.
  availImpl['tree']['DecisionTreeRegressor' ] = (tree.DecisionTreeRegressor , 'float'  )# A tree regressor.
  availImpl['tree']['ExtraTreeClassifier'   ] = (tree.ExtraTreeClassifier   , 'integer')# An extremely randomized tree classifier.
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

  def __init__(self,**kwargs):
    superVisedLearning.__init__(self,**kwargs)
    self.printTag = returnPrintTag('SCIKITLEARN')
    if 'SKLtype' not in self.initOptionDict.keys(): raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> to define a scikit learn ROM the SKLtype keyword is needed (from ROM '+self.name+')')
    SKLtype, SKLsubType = self.initOptionDict['SKLtype'].split('|')
    self.initOptionDict.pop('SKLtype')
    if not SKLtype in self.__class__.availImpl.keys(): raise IOError (self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> not known SKLtype ' + SKLtype +'(from ROM '+self.name+')')
    if not SKLsubType in self.__class__.availImpl[SKLtype].keys(): raise IOError (self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> not known SKLsubType ' + SKLsubType +'(from ROM '+self.name+')')
    self.__class__.returnType     = self.__class__.availImpl[SKLtype][SKLsubType][1]
    self.ROM                      = self.__class__.availImpl[SKLtype][SKLsubType][0]()
    self.__class__.qualityEstType = self.__class__.qualityEstTypeDict[SKLtype][SKLsubType]
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
    @ In, prediction request, Not used in this method (kept the consistency with evaluate method)
    """
    return self.myNumber

  def _readdressEvaluateRomResponse(self,edict):
    """
    Method to re-address the evaluate base class method to its original method
    @ In, prediction request, used in this method (kept the consistency with evaluate method)
    """
    return self.__class__.evaluate(self,edict)

  def __trainLocal__(self,featureVals,targetVals):
    """
    Perform training on samples in featureVals with responses y.
    For an one-class model, +1 or -1 is returned.
    Parameters
    ----------
    featureVals : {array-like, sparse matrix}, shape = [n_samples, n_features]
    Returns
    -------
    targetVals : array, shape = [n_samples]
    """
    #If all the target values are the same no training is needed and the moreover the self.evaluate could be re-addressed to this value
    #print(self.ROM.__dict__)
    #print(self.ROM)
    if len(np.unique(targetVals))>1:
      self.ROM.fit(featureVals,targetVals)
      self.evaluate = self._readdressEvaluateRomResponse
      #self.evaluate = lambda edict : self.__class__.evaluate(self,edict)
    else:
      self.myNumber = np.unique(targetVals)[0]
      self.evaluate = self._readdressEvaluateConstResponse

  def __confidenceLocal__(self,edict):
    if  'probability' in self.__class__.qualityEstType: return self.ROM.predict_proba(edict)
    else            : raise IOError(self.printTag + ': ' +returnPrintPostTag('ERROR') + '-> the ROM '+str(self.name)+'has not the an method to evaluate the confidence of the prediction')

  def __evaluateLocal__(self,featureVals):
    return self.ROM.predict(featureVals)

  def __resetLocal__(self):
    self.ROM = self.ROM.__class__(**self.initOptionDict)

  def __returnInitialParametersLocal__(self):
    return self.ROM.get_params()

  def __returnCurrentSettingLocal__(self):
    print(self.printTag + ': FIXME -> here we need to collect some info on the ROM status')
    localInitParam = {}
    return localInitParam
#
#
#
__interfaceDict                         = {}
__interfaceDict['NDspline'            ] = NDsplineRom
__interfaceDict['NDinvDistWeigth'     ] = NDinvDistWeigth
__interfaceDict['microSphere'         ] = NDmicroSphere
__interfaceDict['SciKitLearn'         ] = SciKitLearn
__interfaceDict['GaussPolynomialRom'] = GaussPolynomialRom
__base                                  = 'superVisedLearning'

def addToInterfaceDict(newDict):
  for key,val in newDict.items():
    __interfaceDict[key]=val

def returnInstance(ROMclass,**kwargs):
  """This function return an instance of the request model type"""
  try: return __interfaceDict[ROMclass](**kwargs)
  except KeyError: raise NameError('not known '+__base+' type '+str(ROMclass))

def returnClass(ROMclass):
  """This function return an instance of the request model type"""
  try: return __interfaceDict[ROMclass]
  except KeyError: raise NameError('not known '+__base+' type '+ROMclass)
