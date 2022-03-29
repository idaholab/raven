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
  Created on August 09, 2021

  @author: Dongli Huang

  This class is for the algorithms of Physics-guided Coverage Mapping
  It inherits from the PostProcessor directly
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import xarray as xr
from scipy import stats
from sklearn.linear_model import OrthogonalMatchingPursuit
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import InputData, InputTypes, randomUtils
from .. import ValidationBase
#Internal Modules End--------------------------------------------------------------------------------

class PhysicsGuidedCoverageMapping(ValidationBase):
  """
    PCM class.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(PhysicsGuidedCoverageMapping, cls).getInputSpecification()
    measurementsInput = InputData.parameterInputFactory("Measurements", contentType=InputTypes.StringListType)
    specs.addSub(measurementsInput)
    return specs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'POSTPROCESSOR PhysicsGuidedCoverageMapping'
    self.name = 'PhysicsGuidedCoverageMapping'
    # Number of bins in KDE
    self.binKDE = 200j

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    # Get Measurements data from input
    for child in paramInput.subparts:
      if child.getName() == 'Measurements':
        self.measurements = child.value
    # Number of Features responses must equal to number of Measurements responses
    # Number of samples between Features and Measurements can be different
    if len(self.features) != len(self.measurements):
      self.raiseAnError(IOError, 'The number of variables found in XML node "Features" is not equal the number of variables found in XML node "Measurements"')
    
  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject
      @ In, inputIn, list, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    dataDict = {self.getDataSetName(data): data for _, _, data in inputIn['Data']}
    names = [self.getDataSetName(inp[-1]) for inp in inputIn['Data']]
    evaluation ={k: np.atleast_1d(val) for k, val in self._evaluate(dataDict, **{'dataobjectNames': names}).items()}
    return evaluation

  def _evaluate(self, datasets, **kwargs):
    """
      Main method to "do what you do".
      @ In, datasets, list, list of datasets (data1,data2,etc.) to used.
      @ In, kwargs, dict, keyword arguments
      @ Out, outputDict, dict, dictionary containing the results {"feat"_"target"_"metric_name":value}
    """
    names = kwargs.get('dataobjectNames')
    outputDict = {}

    # Create an empty array with the same dimension as Target
    featData = np.array([])
    msrData = np.array([])
    featPW = np.array([])
    msrPW = np.array([])    
    numFeats = len(self.features)
    for feat, msr in zip(self.features, self.measurements):
      featDataProb = self._getDataFromDataDict(datasets, feat, names)
      msrDataProb = self._getDataFromDataDict(datasets, msr, names)
      # M>=1 Feature arrays (1D) to 2D array with dimension (N, M)
      featData = np.concatenate((featData, featDataProb[0].reshape(featDataProb[0].shape[0])))
      msrData = np.concatenate((msrData, msrDataProb[0].reshape(msrDataProb[0].shape[0])))
      # Probability Weights for future use
      featPW = np.concatenate((featPW, featDataProb[1]))
      msrPW = np.concatenate((msrPW, msrDataProb[1]))
    featData = featData.reshape(numFeats, -1).T
    msrData = msrData.reshape(numFeats, -1).T 
    featPW = featPW.reshape(numFeats, -1).T
    msrPW = msrPW.reshape(numFeats, -1).T

    # For each Target/Application model/response, calculate an uncertainty reduction fraction
    # using all available Features/Experiments
    stdReduct = []
    for targ in self.targets:
      targDataProb = self._getDataFromDataDict(datasets, targ, names)
      # Data values in <x>Data, <x>=targ, feat, msr
      targData = targDataProb[0]
      # Probability Weights values in <x>PW, , <x>=targ, feat, msr
      targPW = targDataProb[1]

      # Probability Weights to be used in the future
      yExp = np.array(featData)
      yApp = np.array(targData)
      yMsr = np.array(msrData)
      # Sample mean as reference value for simplicity
      # Can be user-defined in the future
      yExpRef = np.mean(yExp, axis=0)
      yAppRef = np.mean(yApp)
      # Usually the reference value is given,
      # and will not be zero, e.g. reference fuel temperature.
      # Standardization
      yExpStd = (yExp-yExpRef)/yExpRef
      yAppStd = (yApp-yAppRef)/yAppRef
      yMsrStd = (yMsr-yExpRef)/yExpRef

      # Single Experiment response
      if yExpStd.shape[1]==1:
        yExpReg = yExpStd.flatten()
        yMsrStd = yMsrStd.flatten()
      # Pseudo response of multiple Experiment responses
      # OrthogonalMatchingPursuit from sklearn used here
      # Possibly change to other regressors      
      elif yExpStd.shape[1]>1:
        regrExp = OrthogonalMatchingPursuit(fit_intercept=False).fit(yExpStd, yAppStd)
        yExpReg = regrExp.predict(yExpStd)
        # Combine measurements by multiple Experiment regression      
        yMsrStd = regrExp.predict(yMsrStd)

      # Kernel Desnity Estimation
      m1 = yExpReg[:]
      m2 = yAppStd.flatten()
      xmin = m1.min()
      xmax = m1.max()
      ymin = m2.min()
      ymax = m2.max()

      # Grid of Experiment (X), grid of Application (Y)
      X, Y = np.mgrid[xmin:xmax:self.binKDE, ymin:ymax:self.binKDE]
      psts = np.vstack([X.ravel(), Y.ravel()])
      vals = np.vstack([m1, m2])
      randomUtils.randomSeed(0)
      # Check whether the covavariance matrix is positive definite
      if np.linalg.det(np.cov(vals))>1e-16:
        # Kernel
        # Need to consider probability weights in the future
        knl = stats.gaussian_kde(vals)
      # If not, introduce a 
      else:
        vals += 1e-5*randomUtils.randomNormal(size=(vals.shape[0], vals.shape[1]))
        knl = stats.gaussian_kde(vals)      
      # Joint PDF of Experiment and Application
      Z = np.reshape(knl(psts).T, X.shape) 

      # Measurement PDF with KDE
      knlMsr = stats.gaussian_kde(yMsrStd)
      pdfMsr = knlMsr(X[:, 0])

      # yAppPred by integrating p(yexp, yapp)p(ymsr) over [yexp.min(), yexp.max()]
      pdfAppPred = np.dot(Z, pdfMsr.reshape(pdfMsr.shape[0], 1))
      # Normalized PDF of predicted application  
      pdfAppPredNorm = pdfAppPred.flatten()/pdfAppPred.sum()/np.diff(Y[0, :])[0]

      # Calculate Expectation (average value) of predicted application
      # by integrating xf(x), where f(x) is PDF of x
      predMean = 0.0
      for i in range(len(Y[0, :])):
        predMean += Y[0, i]*pdfAppPredNorm[i]*(Y[0, 1]-Y[0, 0])

      # Calculate Variance of predicted application
      # by integrating (x-mu_x)^2f(x), where f(x) is PDF of x
      predVar = 0.0
      for i in range(len(Y[0, :])):
        predVar += (Y[0, i]-predMean)**2.0 * pdfAppPredNorm[i]*(Y[0, 1]-Y[0, 0])

      # Predicted standard deviation is square root of variance
      predStd = np.sqrt(predVar)
      # Prior standard deviation is the sample standard deviation
      # Consider probability weights in the future
      priStd = np.std(yAppStd)
      # Uncertainty reduction fraction is 1.0-sigma_pred/sigma_pri
      stdReduct.append(1.0-predStd/priStd)
    stdReduct = np.array(stdReduct)

    name = "pri_post_stdReduct"
    outputDict[name] = stdReduct

    return outputDict
