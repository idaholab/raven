'''
Created on Mar 16, 2013

@author: crisr
'''
from sklearn import svm
import Datas
import numpy
from itertools import product as itprod
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
  def train(self):
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
    #TODO can I accept distDict here?
  def train(self,solns,distDict):
    # solns is dict of soln values keyed on qps
    # distDict is dict of distributions for each variable keyed on var
        # distDict can be reconstructed using:
        # - distro types
        # - distro inputs (mean, std_dev, low, high, alpha, beta, etc)

    qps={} #from hdf5 file, qps indexed on...orders?
    wts={} #from hdf5 file, wts indexed on same?
    # use hdf5 database to link up qps and weights in dict
    qp2wt={}

    self.poly_coeffs={}
    # loop over all possible combinations of expansion orders in each var
    for ords in list(product(*[range(distDict[var].polyOrder()) for var in distDict.keys()])):
      self.poly_coeffs[ords]=0
      for qp in qps:
        poly=wt=probNorm=1.
        for v,var in enumerate(distDict):
          poly*=distDict[var].quad().evNormPoly(ords[v],qp[v])
          wt*=distDict[var].standardToActualWeight(wts[v])
          #TODO assumes standardToActualWeight is a linear transformation!
          probNorm*=distDict[var].probNorm(qp[v])
        self.poly_coeffs[ords]+=solns[qp]*wt*poly*probNorm
  def evaluate(self,valDict,distDict):
    # valDict is dict of values to evaluate at, keyed on var
    # distDict is dict of distributions keyed on var
    tot=0
    for ords,coeff in self.poly_coeff:
      tot+=coeff*np.prod([distDict[var].quad().evNormPoly(\
              ords[v],distDict[var].revertPt(valDict[var])) for v,var in enumerate(valDict)])
      #TODO revertPt may not always be straightforward to implement!
    return tot
  def reset(self):
    #TODO what do I need to do here?
    pass




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
  
