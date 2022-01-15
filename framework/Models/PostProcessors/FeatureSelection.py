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

@author: alfoa
"""
from __future__ import division, print_function , unicode_literals, absolute_import

#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
from sklearn.feature_selection import RFE, RFECV, mutual_info_regression,  mutual_info_classif,  VarianceThreshold
from sklearn.decomposition import KernelPCA,  PCA, FastICA
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
import pandas as pd
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from utils import InputData, InputTypes
import Files
from SupervisedLearning import factory as romFactory
#Internal Modules End--------------------------------------------------------------------------------


class FeatureSelection(PostProcessorInterface):
  """
    Feature Selection PostProcessor is aimed to select the most important features with different methods.
    The importance ranking is provided.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    ## This will replace the lines above
    inputSpecification = super(FeatureSelection, cls).getInputSpecification()
    whatInput = InputTypes.makeEnumType("whatType","whatTypeType",["RFE","RFECV", "mutualInformation", "PCARFE"])
    inputSpecification.addSub(InputData.parameterInputFactory("what", contentType=whatInput))
    TargetsInput = InputData.parameterInputFactory("targets", contentType=InputTypes.StringType)
    inputSpecification.addSub(TargetsInput)
    FeaturesInput = InputData.parameterInputFactory("features", contentType=InputTypes.StringType)
    inputSpecification.addSub(FeaturesInput)    
    PivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType)
    inputSpecification.addSub(PivotParameterInput)
    numberOfFeatures = InputData.parameterInputFactory("minimumNumberOfFeatures", contentType=InputTypes.IntegerType)
    inputSpecification.addSub(numberOfFeatures)
    step = InputData.parameterInputFactory("step", contentType=InputTypes.FloatOrIntType)
    inputSpecification.addSub(step)
    aTarg = InputData.parameterInputFactory("aggregateTargets", contentType=InputTypes.BoolType)
    inputSpecification.addSub(aTarg)
    corrS = InputData.parameterInputFactory("correlationScreening", contentType=InputTypes.BoolType)
    inputSpecification.addSub(corrS)
    ROMInput = InputData.parameterInputFactory("ROM", contentType=InputTypes.StringType)
    ROMInput.addParam("class", InputTypes.StringType)
    ROMInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(ROMInput)    
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.targets = [] # targets
    self.features = None
    self.what = None  # how to perform the selection (list is in InputData specification)
    self.settings = {}
    self.dynamic  = False # is it time-dependent?
    self.pivotParameter = 'Time'
    self.addAssemblerObject('ROM', InputData.Quantity.zero_to_one)
    self.printTag = 'POSTPROCESSOR FEATURE SELECTION'

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    for child in paramInput.subparts:
      if child.getName() == 'ROM':
        continue
      elif child.getName() == 'what':
        self.what = child.value.strip()
      elif child.getName() == 'targets':
        self.targets = list(inp.strip() for inp in child.value.strip().split(','))
      elif child.getName() == 'features':
        self.features = list(inp.strip() for inp in child.value.strip().split(','))      
      elif child.getName() == 'step':
        self.settings[child.getName()] = child.value
      elif child.getName() == 'minimumNumberOfFeatures':
        self.settings[child.getName()] = child.value
      elif child.getName() == 'aggregateTargets':
        self.settings[child.getName()] = child.value
      elif child.getName() == 'correlationScreening':
        self.settings[child.getName()] = child.value
      elif child.getName() == 'pivotParameter':
        self.pivotParameter = child.value
      else:
        self.raiseAnError(IOError, 'Unrecognized xml node name: ' + child.getName() + '!')

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the LS pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    if 'ROM' not in self.assemblerDict.keys():
      self.ROM = romFactory.returnInstance('LinearRegression')
      paramDict = {'Features':list(self.parameters['targets']), 'Target':[self.externalFunction.name]}
      self.ROM.initializeFromDict(paramDict)
      settings = {"n_neighbors":1}
      self.ROM.initializeModel(settings)
    else:
      self.ROM = self.assemblerDict['ROM'][0][3]
    # now we create a wrapper here that can work with scikitlearn 
    class scikitLearnWrapper(BaseEstimator):
      def __init__(self, ROM):
          self.ROM = ROM
  
      def fit(self, X):
          self.n_samples_fit_ = X.shape[0]
          self.annoy_ = annoy.AnnoyIndex(X.shape[1], metric=self.metric)
          for i, x in enumerate(X):
              self.annoy_.add_item(i, x.tolist())
          self.annoy_.build(self.n_trees)
          return self
  
      def transform(self, X):
          return self._transform(X)
  
      def fit_transform(self, X, y=None):
          return self.fit(X)._transform(X=None)
  
      def _transform(self, X):
          """As `transform`, but handles X is None for faster `fit_transform`."""
  
          n_samples_transform = self.n_samples_fit_ if X is None else X.shape[0]
  
          # For compatibility reasons, as each sample is considered as its own
          # neighbor, one extra neighbor will be computed.
          n_neighbors = self.n_neighbors + 1
  
          indices = np.empty((n_samples_transform, n_neighbors), dtype=int)
          distances = np.empty((n_samples_transform, n_neighbors))
  
          if X is None:
              for i in range(self.annoy_.get_n_items()):
                  ind, dist = self.annoy_.get_nns_by_item(
                      i, n_neighbors, self.search_k, include_distances=True
                  )
  
                  indices[i], distances[i] = ind, dist
          else:
              for i, x in enumerate(X):
                  indices[i], distances[i] = self.annoy_.get_nns_by_vector(
                      x.tolist(), n_neighbors, self.search_k, include_distances=True
                  )
  
          indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
          kneighbors_graph = csr_matrix(
              (distances.ravel(), indices.ravel(), indptr),
              shape=(n_samples_transform, self.n_samples_fit_),
          )
  
          return kneighbors_graph
      
    

  def collectOutput(self,finishedJob, output):
    """
      Function to place all of the computed data into the output object, (Files or DataObjects)
      @ In, finishedJob, object, JobHandler object that is in charge of running this postprocessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    outputDict = evaluation[1]
    # Output to DataObjects
    if output.type in ['PointSet','HistorySet']:
      self.raiseADebug('Dumping output in data object named ' + output.name)
      output.load(outputDict, style="dict")
    else:
      self.raiseAnError(IOError, 'Output type ' + str(output.type) + ' unknown.')

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is understandable by this pp.
      @ In, currentInput, object, an object that needs to be converted
      @ Out, inputList, list, list of input dictionaries
    """
    currentInput = currentInp[-1]  if type(currentInp) == list else currentInp
    if type(currentInput) == dict:
      if 'targets' not in currentInput.keys() and 'timeDepData' not in currentInput.keys():
        self.raiseAnError(IOError, 'Did not find targets or timeDepData in input dictionary')
      return currentInput
    if not hasattr(currentInput,'type'):
      self.raiseAnError(IOError, self, 'FeatureSelection postprocessor accepts DataObject(s) only! Got ' + str(type(currentInput)))
    if currentInput.type not in ['PointSet','HistorySet']:
      self.raiseAnError(IOError, self, 'FeatureSelection postprocessor accepts DataObject(s) only! Got ' + str(currentInput.type) + '!!!!')
    # get input from PointSet DataObject
    if currentInput.type in ['PointSet','HistorySet']:
      dataSet = currentInput.asDataset()
      inputDict = {'targets':{}, 'metadata':{}, 'features':{}}
      if self.features is None:
        self.features = list(set(list(dataSet.keys())) - set(list(self.targets)) - set(list(self.pivotParameter))) 
      if currentInput.type in ['HistorySet']:
        self.dynamic = True
        if self.pivotParameter in dataSet.keys():
          self.pivotValue = dataSet[self.pivotParameter].values  
        else:
          self.raiseAnError(IOError, self, 'Time-dependent FeatureSelection is requested (HistorySet) but no pivotParameter got inputted or not in dataset!')
      for feat in self.features:
        inputDict['features'][feat] = copy.copy(dataSet[feat].values)
      for targetP in self.targets:
        if targetP in currentInput.getVars('output'):
          inputDict['targets'][targetP] = copy.copy(dataSet[targetP].values)
        else:
          self.raiseAnError(IOError,'Parameter ' + str(targetP) + ' is listed FeatureSelection postprocessor targets, but not found in the provided output!')
      inputDict['metadata'] = currentInput.getMeta(pointwise=True)
      inputList = [inputDict]
    # get input from HistorySet DataObject
    else:
      dataSet = currentInput.asDataset()
      if self.pivotParameter is None:
        self.raiseAnError(IOError, self, 'Time-dependent FeatureSelection is requested (HistorySet) but no pivotParameter got inputted!')
      self.dynamic = True
      self.pivotValue = dataSet[self.pivotParameter].values
      if not currentInput.checkIndexAlignment(indexesToCheck=self.pivotParameter):
        self.raiseAnError(IOError, "In data object ", currentInput.name, ", the realization' pivot parameters have unsynchronized pivot values!"
                + "Please use the internal postprocessor 'HistorySetSync' to synchronize the data.")
      slices = currentInput.sliceByIndex(self.pivotParameter)
      metadata = currentInput.getMeta(pointwise=True)
      inputList = []
      for sliceData in slices:
        inputDict = {}
        inputDict['metadata'] = metadata
        inputDict['targets'] = dict((target, sliceData[target].values) for target in self.targets)
        inputDict['features'] = dict((feature, sliceData[feature].values) for feature in list(set(list(dataSet.keys())) - set(list(self.targets))))
        inputList.append(inputDict)

    return inputList

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    dataSet = inputIn[0].asDataset()
    #self.__runLocal(dataSet)
    
    inputList = self.inputToInternal(inputIn)
    if not self.dynamic:
      outputDict = self.__runLocal(inputList[0])
    else:
      outputList = []
      for inputDict in inputList:
        outputList.append(self.__runLocal(inputDict))
      outputDict = dict((var, list()) for var in outputList[0].keys())
      for output in outputList:
        for var, value in output.items():
          outputDict[var] = np.append(outputDict[var], value)
      # add the pivot parameter and its values
      outputDict[self.pivotParameter] = np.atleast_1d(self.pivotValue)

    return outputDict

  def __runLocal(self, inputDict):
    """
      This method executes the postprocessor action.
      @ In, inputDict, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, dictionary containing the evaluated data
    """
    # lr = LinearRegression().fit(sampledFeatMatrix, inputDict['targets'][target])
    # compute sensitivities of targets with respect to features
    step = self.settings.get('step', 1)
    nFeatures = self.settings.get('minimumNumberOfFeatures', None)
    aggregateTargets = self.settings.get("aggregateTargets", False)
    correlationScreening = self.settings.get("correlationScreening", False)
    if correlationScreening:
      # Create correlation matrix
      df = pd.DataFrame.from_dict(inputDict['features'])
      # Create correlation matrix
      corr_matrix = df.corr().abs()
      # Select upper triangle of correlation matrix
      upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
      # Find features with correlation greater than 0.95
      to_drop = [column for column in upper.columns if any(upper[column] > 0.9999)]
      inputDict['features'] =  {key:inputDict['features'][key] for key in list(set(list( inputDict['features'].keys())) - set(to_drop))}
      # corrwithtarget = pd.DataFrame.from_dict(a).corr().abs()
      if nFeatures > len(inputDict['features']):
        self.raiseAWarning("number of features selected via correlation analysis is < minimumNumberOfFeatures!")
      nFeatures = min(nFeatures, len(inputDict['features']))

    #kpca = PCA(n_components=10)
    #newFeatures = kpca.fit_transform(np.atleast_2d(list(inputDict['features'].values())).T)    
    #headers = ["PCA_" + str(i+1) for i in range(10)] +  list(inputDict['targets'].keys())
  
    #np.savetxt("pca_10components.csv", np.concatenate((newFeatures, np.atleast_2d(list(inputDict['targets'].values())).T), axis=1), delimiter=',', header=','.join(headers))
    
    #kpca = KernelPCA(n_components=10, kernel = "rbf", random_state=0)
    #newFeatures = kpca.fit_transform(np.atleast_2d(list(inputDict['features'].values())).T)
    
    #np.savetxt("kernelpca_10components.csv", np.concatenate((newFeatures, np.atleast_2d(list(inputDict['targets'].values())).T), axis=1), delimiter=',', header=','.join(headers))
     
    
    #fica = FastICA(n_components=10)
    #newFeatures = fica.fit_transform(np.atleast_2d(list(inputDict['features'].values())).T)
  
    #np.savetxt("ica_10components.csv", np.concatenate((newFeatures, np.atleast_2d(list(inputDict['targets'].values())).T), axis=1), delimiter=',', header=','.join(headers))

    ## 5

    #kpca = PCA(n_components=5)
    #newFeatures = kpca.fit_transform(np.atleast_2d(list(inputDict['features'].values())).T)    
    #headers = ["PCA_" + str(i+1) for i in range(5)] +  list(inputDict['targets'].keys())
  
    #np.savetxt("pca_5components.csv", np.concatenate((newFeatures, np.atleast_2d(list(inputDict['targets'].values())).T), axis=1), delimiter=',', header=','.join(headers))
    
    #kpca = KernelPCA(n_components=5, kernel = "rbf", random_state=0)
    #newFeatures = kpca.fit_transform(np.atleast_2d(list(inputDict['features'].values())).T)
    
    #np.savetxt("kernelpca_5components.csv", np.concatenate((newFeatures, np.atleast_2d(list(inputDict['targets'].values())).T), axis=1), delimiter=',', header=','.join(headers))
     
    
    #fica = FastICA(n_components=5)
    #newFeatures = fica.fit_transform(np.atleast_2d(list(inputDict['features'].values())).T)
  
    #np.savetxt("ica_5components.csv", np.concatenate((newFeatures, np.atleast_2d(list(inputDict['targets'].values())).T), axis=1), delimiter=',', header=','.join(headers))
    
    if self.what == 'PCARFE':
      aggregateTargets = True
    if aggregateTargets:
      # perform a PCA and analyze the first principal component
      kpca = PCA(n_components=1)
      # kpca = KernelPCA(n_components=1, kernel = "rbf", random_state=0)
      newTarget =  kpca.fit_transform(np.atleast_2d(list(inputDict['targets'].values())).T)
    # compute importance rank
    outputDict = {}
    # transformer = FactorAnalysis(n_components=10, random_state=0)
    # tt = transformer.fit(np.atleast_2d(list(inputDict['features'].values())).T)
    if self.what == "RFE":
      selectors = [RFE(self.ROM, n_features_to_select=nFeatures, step=step) for _ in range(len(self.targets))]
      #selectors = [RFE(LinearRegression(), n_features_to_select=nFeatures, step=step) for _ in range(len(self.targets))]
      for i, targ in enumerate(self.targets):
        selectors[i] = selectors[i].fit(np.atleast_2d(list(inputDict['features'].values())).T, newTarget if aggregateTargets else inputDict['targets'][targ])
        self.raiseAMessage("Features downselected to "+str( selectors[i].n_features_) +" for target "+targ)
        outputDict[self.name+"_"+targ] = np.atleast_1d(np.array(list(inputDict['features'].keys()))[selectors[i].support_])
    elif self.what == 'RFECV':
      selectors = [RFECV(LinearRegression(), step=step, min_features_to_select=nFeatures, n_jobs=-1) for _ in range(len(self.targets))]
      minFeaturesSelected = int(1e6)
      for i, targ in enumerate(self.targets):
        selectors[i] = selectors[i].fit(np.atleast_2d(list(inputDict['features'].values())).T, newTarget if aggregateTargets else inputDict['targets'][targ])
        self.raiseAMessage("Features downselected to "+str( selectors[i].n_features_) +" for target "+targ)
        minFeaturesSelected = min(minFeaturesSelected, selectors[i].n_features_)
      for i, targ in enumerate(self.targets):
        outputDict[self.name+"_"+targ] = np.atleast_1d(np.array(list(inputDict['features'].keys()))[selectors[i].support_])[:minFeaturesSelected]
    elif self.what == 'mutualInformation':
      if aggregateTargets:
        sortedFeatures = mutual_info_regression(np.atleast_2d(list(inputDict['features'].values())).T, newTarget).argsort()
        for i, targ in enumerate(self.targets):
          outputDict[self.name+"_"+targ] = np.atleast_1d(np.array(list(inputDict['features'].keys()))[sortedFeatures][-nFeatures:])
      else:
        for i, targ in enumerate(self.targets):
          sortedFeatures = mutual_info_regression(np.atleast_2d(list(inputDict['features'].values())).T, inputDict['targets'][targ]).argsort()
          outputDict[self.name+"_"+targ] = np.atleast_1d(np.array(list(inputDict['features'].keys()))[sortedFeatures][-nFeatures:])
    elif self.what == 'PCARFE':
      kpca = PCA(n_components=min(nFeatures*3, len(inputDict['features'].values())))
      newFeatures = kpca.fit_transform(np.atleast_2d(list(inputDict['features'].values())).T)      
      selectors = RFE(LinearRegression(), n_features_to_select=nFeatures*3, step=step)
      print(newFeatures.shape, newTarget.shape)
      selectors = selectors.fit(np.atleast_2d(newFeatures), newTarget.flatten())
      self.raiseAMessage("Features downselected to "+str(selectors.n_features_))
      print(selectors.support_.shape)
      outputDict[self.name] = newFeatures[:, selectors.support_]
      print(newFeatures[:, selectors.support_].shape)
    
    elif self.what == 'kbest':
      X_new = SelectKBest(f_regression, k=nFeatures).fit_transform(X, y)
      transformer = PCA(n_components=nFeatures, random_state=0).fit(np.atleast_2d(list(inputDict['features'].values()) + list(inputDict['targets'].values())).T)


    return outputDict

