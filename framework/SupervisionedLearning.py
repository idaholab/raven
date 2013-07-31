'''
Created on Mar 16, 2013

@author: crisr
'''
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

from sklearn import svm
import Datas
import numpy
from itertools import product as itprod
import cPickle as pk
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

class superVisioned():
  def __init__(self,**kwargs):
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

class StochasticPolynomials(superVisioned):
  def __init__(self,**kwargs):
    superVisioned.__init__(self,**kwargs)
    print('\n\n...Got to __init__ in StochPolys in SVL...\n')
  def train(self,data):
    print('\n\n...Got to train() in StochPolys in SVL...\n')
    self.solns={}
    
    if data.type=='hdf5':
      attr={'history':None,'prefix':None}
      M=data.returnHistory(attr)
      # copying pattern from OutStreams.py, around line 120
      # FIXME don't have self.toLoadFromList
      endGroupNames = toLoadFromList[0].getEndingGroupNames()
      for index in xrange(len(endGroupNames)):
        #FIXME don't have self.alreadyRead
        if not endGroupNames[index] in self.alreadyRead:
          self.histories[endGroupNames[index]] = self.toLoadFromList[0].returnHistory({'history':endGroupNames[index],'filter':'whole'})
          self.alreadyRead.append(endGroupNames[index])

      self.poly_coeffs={}
      #dictQpCoeffs=pk.load(file('SCweights.pk','r')) take from hdf5
      for ords in dictQpCoeffs.keys():
        self.poly_coeffs[ords]=0
        for qp in dictQpCoeffs[ords].keys():
          self.poly_coeffs[ords]+=dictQpCoeffs[ords][qp]*soln[qp]
    else:
      print('Reading from non-HDF5 for StochPolys not supported yet...')
    return
    # loop over all possible combinations of expansion orders in each var
    #for ords in list(product(*[range(self.distDict[var].polyOrder()) for var in self.distDict.keys()])):
    #  self.poly_coeffs[ords]=0
    #  for qp in quad.indx_ord.keys(): #quadrature points
    #    poly=wt=probNorm=1.
    #    for v,var in enumerate(self.distDict):
    #      poly*=self.distDict[var].quad().evNormPoly(ords[v],qp[v])
    #      wt*=self.distDict[var].standardToActualWeight(qp2wt[qp[v]])
    #      #TODO assumes standardToActualWeight is a linear transformation!
    #      probNorm*=self.distDict[var].probNorm(qp[v])
    #    self.poly_coeffs[ords]+=solns[qp]*wt*poly*probNorm
  def evaluate(self,valDict):
    # valDict is dict of values to evaluate at, keyed on var
    tot=0
    for ords,coeff in self.poly_coeff:
      tot+=coeff*np.prod([self.distDict[var].quad().evNormPoly(\
              ords[v],self.distDict[var].revertPt(valDict[var])) for v,var in enumerate(valDict)])
      #TODO revertPt may not always be straightforward to implement!
    return tot
  def reset(self):
    try:
      del self.poly_coeffs
      del self.distDict
    except: pass




class SVMsciKitLearn(superVisioned):
  def __init__(self,**kwargs):
    superVisioned.__init__(self,**kwargs)
    #dictionary containing the set of available Support Vector Machine by scikitlearn
    self.availSVM = {}
    self.availSVM['LinearSVC'] = svm.LinearSVC
    self.availSVM['C-SVC'    ] = svm.SVC
    self.availSVM['NuSVC'    ] = svm.NuSVC
    self.availSVM['epsSVR'   ] = svm.SVR
    try: self.SVM = self.availSVM[self.initializzationOptionDict['SVMtype']](self.initializzationOptionDict)
    except: raise IOError ('not known support vector machine type')

  def train(self,data):
    """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        class_weight : {dict, 'auto'}, optional
            Weights associated with classes. If not given, all classes
            are supposed to have weight one.
        Returns
        -------
        self : object
            Returns self.
        fit( X, y, sample_weight=None):"""
    self.SVM.fit()

  def returnInitialParamters(self):
    return self.SVM._get_param_names()

  def evaluate(self):
    """Perform regression on samples in X.
        For an one-class model, +1 or -1 is returned.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Returns
        -------
        y_pred : array, shape = [n_samples]
        predict(self, X)"""
    self.SVM.predict()

  def reset(self):
    self.SVM = self.availSVM[self.initializzationOptionDict['SVMtype']](self.initializzationOptionDict)

def returnInstance(Type):
  '''This function return an instance of the request model type'''
  base = 'superVisioned'
  InterfaceDict = {}
  InterfaceDict['SVMscikitLearn'       ] = SVMsciKitLearn
  InterfaceDict['StochasticPolynomials'] = StochasticPolynomials
  try: return InterfaceDict[Type]
  except: raise NameError('not known '+base+' type '+Type)
  
