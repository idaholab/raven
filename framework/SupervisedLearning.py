'''
Created on Mar 16, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

from sklearn import svm
import numpy as np
import Datas
import numpy
import h5py
from itertools import product as itprod
try:
  import cPickle as pk
except ImportError:
  import pickle as pk

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
  def __init__(self,**kwargs):
    self.features = kwargs['featureName']
    self.targets  = kwargs['targetName' ]
    self.initializzationOptionDict = kwargs

  def train(self,obj):
    '''override this method to train the ROM'''
    return

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
    kwargs.pop('targetName')
    kwargs.pop('featureName')
    self.SVM.set_params(**kwargs)

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
    self.SVM.fit(X,y)

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
    pass

class NDsplineRom(NDinterpolatorRom):
  def __init__(self,**kwargs):
    NDinterpolatorRom.__init__(self,**kwargs)
    
    
    #dictionary containing the set of available Support Vector Machine by scikitlearn
    if not self.initializzationOptionDict['SVMtype'] in self.availSVM.keys():
      raise IOError ('not known support vector machine type ' + self.initializzationOptionDict['SVMtype'])
    self.SVM = self.availSVM[self.initializzationOptionDict['SVMtype']]()
    kwargs.pop('SVMtype')
    self.SVM.set_params(**kwargs)

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
    self.SVM.fit(X,y)

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


__interfaceDict                          = {}
__interfaceDict['SVMscikitLearn'          ] = SVMsciKitLearn
__interfaceDict['StochasticPolynomials'   ] = StochasticPolynomials
__interfaceDict['NDspline'                ] = NDsplineRom
__interfaceDict['inverseDistanceWeigthing'] = NDinterpolatorRom  # change when implement the class
__interfaceDict['microSphere'             ] = NDinterpolatorRom # change when implement the class
__base                                   = 'superVisioned'


def returnInstance(Type):
  '''This function return an instance of the request model type'''
  try: return __interfaceDict[Type]
  except KeyError: raise NameError('not known '+__base+' type '+Type)
  
