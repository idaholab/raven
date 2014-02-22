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
from itertools import product as itprod
try:
  import cPickle as pk
except ImportError:
  import pickle as pk
  
if sys.version_info.major > 2:
  import interpolationNDpy3 as interpolationND
else:
  import interpolationNDpy2 as interpolationND  

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

class superVisioned(object):
  '''
  This is the general interface to any supervisioned learning method.
  Essentially it contains a train, and evaluate methods
  '''
  def __init__(self,**kwargs):                      
    if 'Features' not in kwargs.keys(): raise IOError('Super Visioned: Feature names not provided')
    if 'Targets'  not in kwargs.keys(): raise IOError('Super Visioned: Target name not provided')
    self.features = kwargs['Features'].split(',')
    self.target   = kwargs['Targets' ] 
    self.featureValues = None
    self.initializzationOptionDict = kwargs
    self.output = None
    self.targetValues = None

  def train(self,obj):
    '''override this method to train the ROM'''
    inputNames, inputValues  = list(obj.getInpParametersValues().keys()), list(obj.getInpParametersValues().values()) 
    if self.target in obj.getOutParametersValues(): 
      outputValues = obj.getOutParametersValues()[self.target]
    else: raise IOError('The output sought '+self.target+' is not in the training set')
    self.featureValues = np.zeros(shape=(outputValues.size,len(self.features)))
    self.targetValues  = np.zeros(shape=(outputValues.size))
    for feat in self.features:
      if feat not in inputNames: raise IOError('The feature sought '+feat+' is not in the training set')   
      else: self.featureValues[:,inputNames.index(feat)] = inputValues[inputNames.index(feat)][:]
    self.targetValues[:] = outputValues[:]
  
  def prepareInputForPrediction(self,request):
    if len(request)>1: raise IOError('SVM accepts only one input not a list of inputs')
    else: self.request =request[0]
    #first we extract the input names and the corresponding values (it is an implicit mapping)
    if  type(self.request)==str:#one input point requested a as a string
      inputNames  = [entry.split('=')[0]  for entry in self.request.split(',')]
      inputValues = [entry.split('=')[1]  for entry in self.request.split(',')]
    elif type(self.request)==dict:#as a dictionary providing either one or several values as lists or numpy arrays
      inputNames, inputValues = self.request.keys(), self.request.values()
    else:#as a internal data type
      try: #try is used to be sure input.type exist
        print(self.request.type)
        inputNames, inputValues = list(self.request.getInpParametersValues().keys()), list(self.request.getInpParametersValues().values())
      except AttributeError: raise IOError('the request of ROM evaluation is done via a not compatible data')
    #now that the prediction points are read we check the compatibility with the ROM input-output set
    lenght = len(set(inputNames).intersection(self.features))
    if lenght!=len(self.features) or lenght!=len(inputNames):
      raise IOError ('there is a mismatch between the provided request and the ROM structure')
    #build a mapping from the ordering of the input sent in and the ordering inside the ROM
    self.requestToLocalOrdering = []
    for local in self.features:
      self.requestToLocalOrdering.append(inputNames.index(local))
    #building the arrays to send in for the prediction by the ROM
    self.request = np.array([inputValues[index] for index in self.requestToLocalOrdering]).T[0]
    return self.request
 
  def collectOut(self,finishedJob,output,predection):
    '''This method append the ROM evaluation into the output'''
    for feature in self.features:
      output.updateInputValue(feature,self.request[self.features.index(feature)])
    output.updateOutputValue(self.target,predection)

  def reset(self):
    '''override this method to re-instance the ROM'''
    return

  def evaluate(self):
    '''override this method to get the prediction from the ROM'''
    return
    
  def returnInitialParamters(self):
    '''override this method to pass the fix set of parameters of the ROM'''
    InitialParamtersDict={}
    return InitialParamtersDict

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
    self.SVM = self.availSVM[self.initializzationOptionDict['SVMtype']]()
    kwargs.pop('SVMtype')
    kwargs.pop('Targets')
    kwargs.pop('Features')
    self.SVM.set_params(**kwargs)

  def train(self,obj):
    """Perform training on samples in X with responses y.
        For an one-class model, +1 or -1 is returned.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Returns
        -------
        y : array, shape = [n_samples]
    """
    #print("X ",X,"y ",y)
    superVisioned.train(self, obj)
    self.SVM.fit(self.featureValues,self.targetValues)

  def returnInitialParamters(self):
    return self.SVM.get_params()

  def evaluate(self,X):
    """Perform regression on samples in X.
        For an one-class model, +1 or -1 is returned.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Returns
        -------
        y_pred : array, shape = [n_samples]
        predict(self, X)"""
    prediction = self.SVM.predict(X)
    print('SVM           : Prediction by ' + self.initializzationOptionDict['SVMtype'] + '. Predicted value is ' + str(prediction))
    return prediction
  
  def reset(self):
    self.SVM = self.availSVM[self.initializzationOptionDict['SVMtype']](self.initializzationOptionDict)

class NDinterpolatorRom(superVisioned):
  def __init__(self,**kwargs):
    superVisioned.__init__(self,**kwargs)
    self.interpolator = None
    self.initParams   = kwargs
  def train(self,X,y):
    """Perform training on samples in X with responses y.
        For an one-class model, +1 or -1 is returned.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Returns
        -------
        y : array, shape = [n_samples]
    """
    #print("X",X,"y",y)
    self.interpolator.fit(X,y)

  def returnInitialParamters(self):
    return self.initializzationOptionDict

  def evaluate(self,X):
    """Perform regression on samples in X.
        For an one-class model, +1 or -1 is returned.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Returns
        -------
        y_pred : array, shape = [n_samples]
        predict(self, X)"""
    prediction = np.zeros(X[:,0].shape)
    for n_sample in range(X[:,0].shape):
      prediction[n_sample] = self.interpolator.interpolateAt(X[:,n_sample])
      print('NDinterpRom   : Prediction by ' + self.name + '. Predicted value is ' + str(prediction))
    return prediction
  
  def reset(self):
    raise NotImplemented('NDinterpRom   : reset method must be implemented!!!!!')

class NDsplineRom(NDinterpolatorRom):
  def __init__(self,**kwargs):
    NDinterpolatorRom.__init__(self,**kwargs)
    self.interpolator = interpolationND.NDspline()

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
  
