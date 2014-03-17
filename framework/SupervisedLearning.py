'''
Created on Mar 16, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

from sklearn import svm
import sys
import numpy as np
import Datas
import numpy
import h5py
import abc
import ast
from itertools import product as itprod
try:
  import cPickle as pk
except ImportError:
  import pickle as pk
  
if sys.version_info.major > 2:
  import interpolationNDpy3 as interpolationND
else:
  import interpolationNDpy2 as interpolationND  
from utils import metaclass_insert
#import DataBases #TODO shouldn't need this, see StochPoly.train() for instance check
'''here we intend ROM as super-visioned learning, 
   where we try to understand the underlying model by a set of labeled sample
   a sample is composed by (feature,label) that is easy translated in (input,output)
   '''
'''module to be loaded from scikitlearn
 Generalized Linear Models
 Support Vector Machines
 Stochastic Gradient Descent
 Nearest Neighbors
 Gaussian Processes
 Partial Least Squares
 Naive Bayes
 Decision Trees
 Ensemble methods
 Multiclass and multilabel algorithms
 Feature selection
 Linear and Quadratic Discriminant Analysis
 Isotonic regression
 '''

class superVisioned(metaclass_insert(abc.ABCMeta)):
  '''
  This is the general interface to any supervisioned learning method.
  Essentially it contains a train, and evaluate methods
  '''
  @classmethod 
  def checkArrayConsistency(self,arrayin):
    '''
      This method checks the consistency of the in-array
      @ In, object... It should be an array
      @ Out, tuple, tuple[0] is a bool (True -> everything is ok, False -> something wrong), tuple[1], string ,the error mesg
    '''
    if type(arrayin) != numpy.ndarray: return (False,' The object is not a numpy array')
    if len(arrayin.shape) > 1: return(False, ' The array must be 1-d')  
    return (True,'')
  
  def __init__(self,**kwargs):                      
    if 'Features' not in kwargs.keys(): raise IOError('Super Visioned: ERROR -> Feature names not provided')
    if 'Target'   not in kwargs.keys(): raise IOError('Super Visioned: ERROR ->Target name not provided')
    self.features = kwargs['Features'].split(',')
    self.target   = kwargs['Target'  ] 
    if self.features.count(self.target) > 0: raise IOError('Super Visioned: ERROR -> The target and one of the features have the same name!!!!')
    self.initializzationOptionDict = kwargs

  def train(self,tdict):
    '''
      Method to perform the training of the SuperVisioned algorithm
      NB.the SuperVisioned object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires.
      @ In, tdict, training dictionary
      @ Out, None
    '''
    if type(tdict) != dict: raise IOError('Super Visioned: ERROR -> method "train". The training set needs to be provided through a dictionary. Type of the in-object is ' + str(type(tdict)))
    names, values  = list(tdict.keys()), list(tdict.values()) 
    if self.target in names: targetValues = values[names.index(self.target)]  
    else                   : raise IOError('Super Visioned: ERROR -> The output sought '+self.target+' is not in the training set')    
    # check if the targetValues are consistent with the expected structure
    resp = self.checkArrayConsistency(targetValues)
    if not resp[0]: raise IOError('Super Visioned: ERROR -> In training set for target '+self.target+':'+resp[1])
    # construct the evaluation matrix
    featureValues = np.zeros(shape=(targetValues.size,len(self.features))) 
    for cnt, feat in enumerate(self.features):
      if feat not in names: raise IOError('Super Visioned: ERROR -> The feature sought '+feat+' is not in the training set')   
      else: 
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]: raise IOError('Super Visioned: ERROR -> In training set for feature '+feat+':'+resp[1])
        if values[names.index(feat)].size != featureValues[:,0].size: raise IOError('Super Visioned: ERROR -> In training set, the number of values provided for feature '+feat+' are != number of target outcomes!')
        featureValues[:,cnt] = values[names.index(feat)]
    self.__trainLocal__(featureValues,targetValues)

  @abc.abstractmethod
  def __trainLocal__(self,featureVals,targetVals):
    '''
      this method must be overwritten by the base classes
      @ In, featureVals, 2-D numpy array [n_samples,n_features]
      @ In,  targetVals, 1-D numpy arrat [n_samples]
    '''
    pass
  
  def confidence(self,edict):
    if type(edict) != dict: raise IOError('Super Visioned: ERROR -> method "confidence". The inquiring set needs to be provided through a dictionary. Type of the in-object is ' + str(type(edict)))
    names, values   = list(edict.keys()), list(edict.values()) 
    for index in range(len(values)): 
      resp = self.checkArrayConsistency(values[index])
      if not resp[0]: raise IOError('Super Visioned: ERROR -> In evaluate request for feature '+names[index]+':'+resp[1])
    featureValues = np.zeros(shape=(values[0].size,len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names: raise IOError('Super Visioned: ERROR -> The feature sought '+feat+' is not in the evaluate set')   
      else: 
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]: raise IOError('Super Visioned: ERROR -> In training set for feature '+feat+':'+resp[1])
        featureValues[:,cnt] = values[names.index(feat)]
    return self.__confidenceLocal__(featureValues)
    
  def evaluate(self,edict):
    '''
      Method to perform the evaluation of a point or a set of points through the previous trained superVisioned algorithm 
      NB.the SuperVisioned object is committed to convert the dictionary that is passed (in), into the local format
      the interface with the kernels requires.
      @ In, tdict, evaluation dictionary
      @ Out, numpy array of evaluated points
    '''
    if type(edict) != dict: raise IOError('Super Visioned: ERROR -> method "evaluate". The evaluate request/s need/s to be provided through a dictionary. Type of the in-object is ' + str(type(edict)))
    names, values  = list(edict.keys()), list(edict.values()) 
    for index in range(len(values)): 
      resp = self.checkArrayConsistency(values[index])
      if not resp[0]: raise IOError('Super Visioned: ERROR -> In evaluate request for feature '+names[index]+':'+resp[1])
    # construct the evaluation matrix
    featureValues = np.zeros(shape=(values[0].size,len(self.features)))
    for cnt, feat in enumerate(self.features):
      if feat not in names: raise IOError('Super Visioned: ERROR -> The feature sought '+feat+' is not in the evaluate set')   
      else: 
        resp = self.checkArrayConsistency(values[names.index(feat)])
        if not resp[0]: raise IOError('Super Visioned: ERROR -> In training set for feature '+feat+':'+resp[1])
        featureValues[:,cnt] = values[names.index(feat)]
    return self.__evaluateLocal__(featureValues)

  @abc.abstractmethod
  def __evaluateLocal__(self,featureVals):
    '''
      this method must be overwritten by the base classes
      @ In, featureVals, 2-D numpy array [n_samples,n_features]
    '''
    pass

  def reset(self):
    '''override this method to re-instance the ROM'''
    return

  def returnInitialParameters(self):
    '''override this method to pass the fix set of parameters of the ROM'''
    return self.initializzationOptionDict

  def returnCurrentSetting(self):
    '''override this method to pass the set of parameters of the ROM that can change during simulation'''
    CurrentSettingDict={}
    return CurrentSettingDict
#
#
#
class StochasticPolynomials(superVisioned):
  def __init__(self,**kwargs):
    superVisioned.__init__(self,**kwargs)
  
  def __trainLocal__(self,featureVals,targetVals):
    pass
  
  def __evaluateLocal__(self,featureVals):
    pass  

  def train(self,inDictionary):
    pass

  def evaluate(self,valDict):
    pass

  def reset(self,*args):
    pass
#
#
#
class SVMsciKitLearn(superVisioned):
  def __init__(self,**kwargs):
    superVisioned.__init__(self,**kwargs)
    #dictionary containing the set of available Support Vector Machine by scikitlearn
    self.availSVM = {}
    self.availSVM['LinearSVC'] = svm.LinearSVC
    self.availSVM['C-SVC'    ] = svm.SVC
    self.availSVM['NuSVC'    ] = svm.NuSVC
    self.availSVM['epsSVR'   ] = svm.SVR
    if not self.initializzationOptionDict['SVMtype'] in self.availSVM.keys(): raise IOError ('not known support vector machine type ' + self.initializzationOptionDict['SVMtype'])
    kwargs.pop('SVMtype')
    kwargs.pop('Target')
    kwargs.pop('Features')
    self.kwargs = kwargs
    self.SVM = self.availSVM[self.initializzationOptionDict['SVMtype']]()
    if 'probability' not in self.kwargs.keys(): self.kwargs['probability'] = True
    if self.initializzationOptionDict['SVMtype'] == 'LinearSVC': self.kwargs.pop('probability')
    for key,value in self.kwargs.items():
      try:self.kwargs[key] = ast.literal_eval(value)
      except: pass
    self.SVM.set_params(**self.kwargs)

  def __trainLocal__(self,featureVals,targetVals):
    """Perform training on samples in X with responses y.
        For an one-class model, +1 or -1 is returned.
        Parameters
        ----------
        featureVals : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Returns
        -------
        targetVals : array, shape = [n_samples]
    """
    #here first we test if at least two class are present otherwise the classifier is just the value itself
    if len(np.unique(targetVals))>1: 
      self.SVM.fit(featureVals,targetVals)
      self.evaluate = lambda edict : superVisioned.evaluate(self,edict)   #lambda self : myNumber superVisioned.evaluate()
    else:
      myNumber = np.unique(targetVals)[0]
      myReturn = lambda edict : myNumber
      self.evaluate = myReturn

  def returnInitialParamters(self):
    return self.SVM.get_params()

  def __confidenceLocal__(self,edict):
    probability = self.SVM.predict_proba(edict)
#    np.sort(probability)
    #probability = probability[:,0]
    return probability


  def __evaluateLocal__(self,featureVals):
    '''
      Perform regression on samples in featureVals.
        For an one-class model, +1 or -1 is returned.
        @ In, numpy.array 2-D, features 
        @ Out, numpy.array 1-D, predicted values
    '''
    prediction = self.SVM.predict(featureVals)
    print('SVM           : Prediction by ' + self.initializzationOptionDict['SVMtype'] + '. Predicted value is ' + str(prediction[-1]))
    return prediction
  
  def reset(self):
    self.SVM = self.availSVM[self.initializzationOptionDict['SVMtype']](self.initializzationOptionDict)

class NDinterpolatorRom(superVisioned):
  def __init__(self,**kwargs):
    superVisioned.__init__(self,**kwargs)
    self.interpolator = None
    self.initParams   = kwargs
    self.type         = None

  def __trainLocal__(self,featureVals,targetVals):
    """Perform training on samples in X with responses y.
        For an one-class model, +1 or -1 is returned.
        Parameters
        ----------
        featureVals : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Returns
        -------
        targetVals : array, shape = [n_samples]
    """
    featv = interpolationND.vectd2d(featureVals[:][:])
    targv = interpolationND.vectd(targetVals)
    self.interpolator.fit(featv,targv)

  def returnInitialParamters(self):
    return self.initializzationOptionDict

  def __evaluateLocal__(self,featureVals):
    '''
      Perform regression on samples in featureVals.
        For an one-class model, +1 or -1 is returned.
        @ In, numpy.array 2-D, features 
        @ Out, numpy.array 1-D, predicted values
    '''
    prediction = np.zeros(featureVals.shape[0])
    for n_sample in range(featureVals.shape[0]):
      featv = interpolationND.vectd(featureVals[n_sample][:])
      prediction[n_sample] = self.interpolator.interpolateAt(featv)
      print('NDinterpRom   : Prediction by ' + self.type + '. Predicted value is ' + str(prediction[n_sample]))
    return prediction
  
  def reset(self):
    raise NotImplemented('NDinterpRom   : reset method must be implemented!!!!!')

class NDsplineRom(NDinterpolatorRom):
  def __init__(self,**kwargs):
    NDinterpolatorRom.__init__(self,**kwargs)
    self.interpolator = interpolationND.NDspline()
    self.type         = 'NDsplineRom'

  def reset(self):
    '''
      The reset here erase the Interpolator...
      keeping the instance...
    '''
    self.interpolator.reset()

class NDinvDistWeigth(NDinterpolatorRom):
  def __init__(self,**kwargs):
    NDinterpolatorRom.__init__(self,**kwargs)
    if not 'p' in self.initializzationOptionDict.keys(): raise IOError ('NDinvDistWeigth: the <p> parameter must be provided in order to use NDinvDistWeigth as ROM!!!!')
    self.interpolator = interpolationND.inverseDistanceWeigthing(float(self.initializzationOptionDict['p']))
    self.type         = 'NDinvDistWeigth'
  def reset(self):
    '''
      The reset here erase the Interpolator...
      keeping the instance...
    '''
    self.interpolator.reset(float(self.initializzationOptionDict['p']))

class NDmicroSphere(NDinterpolatorRom):
  def __init__(self,**kwargs):
    NDinterpolatorRom.__init__(self,**kwargs)
    if not 'p' in self.initializzationOptionDict.keys(): raise IOError ('NDmicroSphere : the <p> parameter must be provided in order to use NDmicroSphere as ROM!!!!')
    if not 'precision' in self.initializzationOptionDict.keys(): raise IOError ('NDmicroSphere : the <precision> parameter must be provided in order to use NDmicroSphere as ROM!!!!')
    self.interpolator = interpolationND.microSphere(float(self.initializzationOptionDict['p']),int(self.initializzationOptionDict['precision']))
    self.type         = 'NDmicroSphere'
  def reset(self):
    '''
      The reset here erase the Interpolator...
      keeping the instance...
    '''
    self.interpolator.reset(float(self.initializzationOptionDict['p']),int(self.initializzationOptionDict['precision']))



__interfaceDict                          = {}
__interfaceDict['SVMscikitLearn'          ] = SVMsciKitLearn
__interfaceDict['StochasticPolynomials'   ] = StochasticPolynomials
__interfaceDict['NDspline'                ] = NDsplineRom
__interfaceDict['NDinvDistWeigth'         ] = NDinvDistWeigth  
__interfaceDict['microSphere'             ] = NDmicroSphere 
__base                                   = 'superVisioned'


def returnInstance(Type):
  '''This function return an instance of the request model type'''
  try: return __interfaceDict[Type]
  except KeyError: raise NameError('not known '+__base+' type '+Type)
  
