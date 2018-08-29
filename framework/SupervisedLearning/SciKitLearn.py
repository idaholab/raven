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
  Created on May 8, 2018

  @author: talbpaul
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
  Specific ROM implementation for SciKitLearn ROMs
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sklearn
from sklearn import linear_model, svm, multiclass, naive_bayes, neighbors, tree, gaussian_process

if int(sklearn.__version__.split(".")[1]) > 16:
  # FIXME: to be removed when the supported minimum version of sklearn is moved to 0.17
  from sklearn import discriminant_analysis as da
else:
  from sklearn import qda
  from sklearn import lda

# To be removed when the supported minimum version of sklearn is moved to 0.18
if int(sklearn.__version__.split(".")[1]) > 17:
  from sklearn import neural_network

import numpy as np
import ast
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .SupervisedLearning import supervisedLearning
#Internal Modules End--------------------------------------------------------------------------------

class SciKitLearn(supervisedLearning):
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
  availImpl['qda']                                          = {}                                                            #Quadratic Discriminant Analysis
  if int(sklearn.__version__.split(".")[1]) > 16:
    availImpl['lda']['LDA']                                 = (da.LinearDiscriminantAnalysis            , 'int'    , False) #Linear Discriminant Analysis (LDA)
    availImpl['qda']['QDA']                                 = (da.QuadraticDiscriminantAnalysis         , 'int'    , False) #Quadratic Discriminant Analysis (QDA)
  else:
    availImpl['lda']['LDA']                                 = (lda.LDA                                  , 'int'    , False) #Linear Discriminant Analysis (LDA)
    availImpl['qda']['QDA']                                 = (qda.QDA                                  , 'int'    , False) #Quadratic Discriminant Analysis (QDA)

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

  availImpl['tree'] = {}
  availImpl['tree']['DecisionTreeClassifier'              ] = (tree.DecisionTreeClassifier              , 'int'   ,  True )# A decision tree classifier.
  availImpl['tree']['DecisionTreeRegressor'               ] = (tree.DecisionTreeRegressor               , 'float' ,  True )# A tree regressor.
  availImpl['tree']['ExtraTreeClassifier'                 ] = (tree.ExtraTreeClassifier                 , 'int'   ,  True )# An extremely randomized tree classifier.
  availImpl['tree']['ExtraTreeRegressor'                  ] = (tree.ExtraTreeRegressor                  , 'float' ,  True )# An extremely randomized tree regressor.

  availImpl['GaussianProcess'] = {}
  availImpl['GaussianProcess']['GaussianProcess'          ] = (gaussian_process.GaussianProcess         , 'float' ,  False)
  # Neural network models (supervised)
  # To be removed when the supported minimum version of sklearn is moved to 0.18
  if int(sklearn.__version__.split(".")[1]) > 17:
    availImpl['neural_network'] = {}
    availImpl['neural_network']['MLPClassifier'              ] = (neural_network.MLPClassifier             , 'int'   ,  True)  # Multi-layer perceptron classifier.
    availImpl['neural_network']['MLPRegressor'               ] = (neural_network.MLPRegressor              , 'float' ,  True)  # Multi-layer perceptron regressor.

  #test if a method to estimate the probability of the prediction is available
  qualityEstTypeDict = {}
  for key1, myDict in availImpl.items():
    qualityEstTypeDict[key1] = {}
    for key2 in myDict:
      qualityEstTypeDict[key1][key2] = []
      if  callable(getattr(myDict[key2][0], "predict_proba", None)):
        qualityEstTypeDict[key1][key2] += ['probability']
      elif  callable(getattr(myDict[key2][0], "score"        , None)):
        qualityEstTypeDict[key1][key2] += ['score']
      else:
        qualityEstTypeDict[key1][key2] = False

  def __init__(self,messageHandler,**kwargs):
    """
      A constructor that will appropriately initialize a supervised learning object
      @ In, messageHandler, MessageHandler object, it is in charge of raising errors, and printing messages
      @ In, kwargs, dict, an arbitrary list of kwargs
      @ Out, None
    """
    supervisedLearning.__init__(self,messageHandler,**kwargs)
    name  = self.initOptionDict.pop('name','')
    # some keywords aren't useful for this ROM
    if 'pivotParameter' in self.initOptionDict:
      # remove pivot parameter if present
      self.initOptionDict.pop('pivotParameter',None)
    self.initOptionDict.pop('paramInput',None)
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
      if self.intrinsicMultiTarget:
        self.ROM = [self.__class__.availImpl[SKLtype][SKLsubType][0](estimator)]
      else:
        self.ROM = [self.__class__.availImpl[SKLtype][SKLsubType][0](estimator) for _ in range(len(self.target))]
    else:
      if self.intrinsicMultiTarget:
        self.ROM = [self.__class__.availImpl[SKLtype][SKLsubType][0]()]
      else:
        self.ROM = [self.__class__.availImpl[SKLtype][SKLsubType][0]() for _ in range(len(self.target))]

    for key,value in self.initOptionDict.items():
      try:
        self.initOptionDict[key] = ast.literal_eval(value)
      except:
        pass

    for index in range(len(self.ROM)):
      self.ROM[index].set_params(**self.initOptionDict)

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
    #get the number of inputs provided to this ROM to evaluate
    numInputs = len(edict.values()[0])
    #fill the target values
    for index,target in enumerate(self.target):
      returnDict[target] = np.ones(numInputs)*self.myNumber[index]
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
      # if all targets only have a single unique value, just store that value, no need to fit/train
      if all([len(np.unique(targetVals[:,index])) == 1 for index in range(len(self.ROM))]):
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
    else:
      self.raiseAnError(IOError,'the ROM '+str(self.initOptionDict['name'])+'has not the an method to evaluate the confidence of the prediction')
    return confidenceDict

  def __evaluateLocal__(self,featureVals):
    """
      Evaluates a point.
      @ In, featureVals, np.array, list of values at which to evaluate the ROM
      @ Out, returnDict, dict, dict of all the target results
    """
    returnDict = {}
    if not self.intrinsicMultiTarget:
      for index, target in enumerate(self.target):
        returnDict[target] = self.ROM[index].predict(featureVals)
    else:
      outcome = self.ROM[0].predict(featureVals)
      for index, target in enumerate(self.target):
        returnDict[target] = outcome[:,index]
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

