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

  Modified by Shiming Yin on Dec. 02, 2022

  This class is for the algorithms of Physics-guided Coverage Mapping
  It inherits from the PostProcessor directly
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import xarray as xr
from scipy import stats
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ravenframework.utils import InputData, InputTypes
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
    names = list(dataDict.keys())#
    evaluation ={k: np.atleast_1d(val) for k, val in self._evaluate(inputIn['Data'][0][-1].coords.keys(), dataDict, **{'dataobjectNames': names}).items()}
    if 'snapshot_pri_post_stdReduct' in evaluation.keys():
      pivotParameter = self.pivotParameter
      evaluation[pivotParameter] = inputIn['Data'][0][-1]['timeSnapshot']
    if 'Tdep_post_mean' in evaluation.keys():
      pivotParameter = self.pivotParameter
      evaluation[pivotParameter] = inputIn['Data'][0][-1]['timeTdep']
    return evaluation

  def _evaluate(self, keys, datasets, **kwargs):
    """
      Main method of PCM. It collects the response values from Feature and Target models,
      and Measurements from experiment, maps the biases and uncertainties from Feature to
      Target side, and calculates the uncertainty reduction fraction using Feature to
      validate Target.
      @ In, datasets, list, list of datasets (data1,data2,etc.) to used.
      @ In, kwargs, dict, keyword arguments
      @ Out, outputDict, dict, dictionary containing the results {"pri_post_stdReduct_<targName>":value}
    """
    """
      Functions to be applied
    """
    #PCM function for snapshot PCM and Static PCM
    def PCM(featData, msrData, targData):
      outputArray = []
      for t in range(targData.shape[1]):
        # Probability Weights to be used in the future
        yExp = np.array(featData)
        yMsr = np.array(msrData)
        # Reference values of Experiments, yExpRef in M
        # Sample mean as reference value for simplicity
        # Can be user-defined in the future
        yExpRef = np.mean(yExp, axis=0)
        # Usually the reference value is given,
        # and will not be zero, e.g. reference fuel temperature.
        # Standardization
        yExpStd = (yExp-yExpRef)/yExpRef
        yMsrStd = (yMsr-yExpRef)/yExpRef

        # For each Target/Application model/response, calculate an uncertainty reduction fraction
        # using all available Features/Experiments

        # Application responses yApp in Nx1
        yApp = np.array(targData[:,t])
        # Reference values of Application, yAppRef is a scalar
        yAppRef = np.mean(yApp)
        # Standardization
        yAppStd = (yApp-yAppRef)/yAppRef

        # Single Experiment response
        if yExpStd.shape[1]==1:
          yExpReg = yExpStd.flatten()
          yMsrReg = yMsrStd.flatten()
        # Pseudo response of multiple Experiment responses
        # OrthogonalMatchingPursuit from sklearn used here
        # Possibly change to other regressors
        elif yExpStd.shape[1]>1:
          regrExp = OrthogonalMatchingPursuit(fit_intercept=False).fit(yExpStd, yAppStd)
          yExpReg = regrExp.predict(yExpStd)
          # Combine measurements by multiple Experiment regression
          yMsrReg = regrExp.predict(yMsrStd)

        # Measurement PDF with KDE
        knlMsr = stats.gaussian_kde(yMsrReg)

        # KDE for joint PDF between Exp and App
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
        # Measurement PDF over Exp range
        pdfMsr = knlMsr(X[:, 0])

        # Condition number of matrix of feature and target
        condNum = np.linalg.cond(vals)
        # If condition number is greater than 100
        invErr = 100
        # Check whether the covavariance matrix is positive definite
        if condNum>=invErr:
            # If singular matrix, measurement of Experiment is directly transfered
            # as predicted Application
            pdfAppPred = knlMsr(Y[0, :])
        else:
            # If not, KDE of Experiment and Application
            knl = stats.gaussian_kde(vals)
            # Joint PDF of Experiment and Application
            Z = np.reshape(knl(psts).T, X.shape)
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
        outputArray.append(1.0-predStd/priStd)
      return outputArray

    # functions for T-Dep PCM
    def KNN(exp, app, expM, k=5, Metric='minkowski'):
      # check the dimension of exp matrix, \
      # if it only has one feature, enasure to make it as a 2D matrix instead of a 1D vector
      if expM.ndim==1:
          expM = expM.reshape(1, expM.shape[0])
      # Scale exp data with its mean (mu) and standard deviation (std).  [expS = (exp - mu)/std]
      # Scale expM data with by same mu and std. [expMS = (expM - mu)/std]
      scaler = StandardScaler()
      expS = scaler.fit_transform(exp)
      expMS = scaler.transform(expM)
      # Fit KNN regression model
      model = KNeighborsRegressor(n_neighbors=k, metric=Metric)
      model.fit(expS, app)
      # Predict on measurement set
      appPred = model.predict(expMS)

      return appPred

    def FindRank(ref, x, xM):
      x = (x.T/ref).T
      error = (np.std(xM, axis=0)/ np.mean(xM, axis=0))[0] #msr error
      #scaling by reference
      u, s, v_T = np.linalg.svd(x)
      # Find the maximum of the residual under different rank
      maxError = np.zeros(np.mean(xM, axis=0).shape[0])
      for k in range(1,maxError.shape[0]+1):
        res = x - u[:,:k]@np.diag(s)[:k,:k]@v_T[:k,:]#residual under different rank
        maxError[k-1] = np.max(abs(res)) # maximum of the residual
      xMAX=min(np.argwhere(maxError<error/6)) #rank corresponding to msr error

      return xMAX[0]

    # Find a reference sample which is closest to samples' average for scaling later
    def findRef(yExp, yApp):
      # calculate average temperatures among samples (samples' average)
      yExpMean = np.mean(yExp, axis=0)
      yAppMean = np.mean(yApp, axis=0)
      # find the reference that is closest to the mean by employing Mean Square Error as metric
      mse1 = np.zeros(yExp.shape[0])
      mse2 = np.zeros(yExp.shape[0])
      for i in range(yExp.shape[0]):
          mse1[i] = np.mean((yExp[i,:]-yExpMean)**2)
          mse2[i] = np.mean((yApp[i,:]-yAppMean)**2)
      # The index for the reference sample
      inExp = np.where(mse1==np.min(mse1))[0][0]
      inApp = np.where(mse2==np.min(mse2))[0][0]
      return inExp, inApp

    def pcmTdep(featData, msrData, targData):
      yExp = np.array(featData)
      yApp = np.array(targData)
      yExpMsr = np.array(msrData)
      yExpMsrMean = np.mean(np.array(msrData), axis=0)

      # calculate alphaApp as linear combination of alphaExp (with RankApp, RankExp)
      inExp, inApp = findRef(yExp, yApp)
      yAppRef = yApp[inApp, :]
      yAppScaled = (np.delete(yApp[:,:] - yAppRef, inApp,0)).T #scaled application prior data
      uApp, sApp, vApp_T = np.linalg.svd(yAppScaled, full_matrices=False)
      alphaApp = (uApp.T@yAppScaled).T # coefficients of application prior data under UApp subspace


      yExpRef = yExp[inExp, :]
      yExpScaled = (np.delete(yExp[:,:] - yExpRef, inExp,0)).T #scaled experiment prior data
      uExp, sExp, vExp_T = np.linalg.svd(yExpScaled, full_matrices=False)
      alphaExp = (uExp.T@yExpScaled).T # coefficients of experiment prior data under UExp subspace

      yExpMsrScaled = ((yExpMsrMean - yExpRef)).T # scaled experiments' Measurement data
      alphaExpMsr = (uExp.T@yExpMsrScaled).T #  coefficients of experiments' Measurement data under UExp subspace


      #KNN regression
      rkApp = FindRank(yAppRef, yAppScaled, yExpMsr) #rank of application
      rkExp = FindRank(yExpRef, yExpScaled, yExpMsr)  # rank of experiments
      alphaAppHat=np.zeros(rkApp)
      for i in range(rkApp):
        alphaAppHat[i] = KNN(alphaExp[:,:rkExp], alphaApp[:,i], alphaExpMsr[:rkExp],k=4)

      # reconstruct posterior appliaction data by the perdicted application coefficients alphaAppm_hat
      yAppPredScaled = (uApp[:,:rkApp] @ alphaAppHat.T).T
      yAppPred = yAppPredScaled + yAppRef
      error = abs(yAppPred-yAppRef)/yAppRef
      return yAppPred, error


    """
      Data reading and processing with different version of PCM
    """
    names = kwargs.get('dataobjectNames')
    outputDict = {}
    # Create empty list for multiple Exp responses
    featData = []
    msrData = []
    targData =[]
    featPW = []
    msrPW = []

    for feat, msr, targ in zip(self.features, self.measurements, self.targets):
      featDataProb = self._getDataFromDataDict(datasets, feat, names)
      msrDataProb = self._getDataFromDataDict(datasets, msr, names)
      # read targets' data
      targDataProb = self._getDataFromDataDict(datasets, targ, names)
      # M>=1 Feature arrays (1D) to 2D array with dimension (N, M)
      # Data values in <x>Data, <x>=targ, feat, msr
      featData.append(featDataProb[0].flatten())
      msrData.append(msrDataProb[0].flatten())
      targData.append(targDataProb[0].flatten())

    # Probability Weights for future use
    featPW.append(featDataProb[1])
    msrPW.append(msrDataProb[1])
    # Probability Weights values in <x>PW, , <x>=targ, feat, msr
    targPW = targDataProb[1]

    featPW = np.array(featPW).T
    msrPW = np.array(msrPW).T
    targPW = np.array(targPW).T

    if 'timeTdep' in keys:
      pcmVersion = 'Tdep'
      self.raiseAMessage('***    Running Tdep-PCM       ***')
      # Data of size (num_of_samples, num_of_features)
      element = np.asarray(datasets['exp'].get('timeTdep'))[0]
      v = np.asarray(datasets['exp'].get('timeTdep'))
      num_of_samples = np.count_nonzero(v == element)
      num_of_featuresExp = int(np.asarray(datasets['exp'].get('timeTdep')).shape[0]/num_of_samples)
      num_of_featuresApp = int(np.asarray(datasets['app'].get('timeTdep')).shape[0]/num_of_samples)
      featData = np.array(featData).reshape(num_of_samples, num_of_featuresExp)
      msrData = np.array(msrData).reshape(num_of_samples, num_of_featuresExp)
      targData = np.array(targData).reshape(num_of_samples, num_of_featuresApp)
      time = v[:num_of_featuresExp]
      yAppPred, error = pcmTdep(featData, msrData, targData)

    elif 'timeSnapshot' in keys:
      pcmVersion = 'snapshot'
      self.raiseAMessage('***    Running Snapshot-PCM   ***')
      # Data of size (num_of_samples, num_of_features)
      element = np.asarray(datasets['exp'].get('timeSnapshot'))[0]
      v = np.asarray(datasets['exp'].get('timeSnapshot'))
      num_of_samples = np.count_nonzero(v == element)
      num_of_features = int(np.asarray(datasets['exp'].get('timeSnapshot')).shape[0]/num_of_samples)
      num_of_features = int(np.asarray(datasets['app'].get('timeSnapshot')).shape[0]/num_of_samples)
      featData = np.array(featData).reshape(num_of_samples, num_of_features)
      msrData = np.array(msrData).reshape(num_of_samples, num_of_features)
      targData = np.array(targData).reshape(num_of_samples, num_of_features)
      time = v[:num_of_features]
      outputArray = PCM(featData, msrData, targData)

    else:
      pcmVersion = 'static'
      self.raiseAMessage('***     Running Static-PCM    ***')
      featData = np.array(featData).T
      msrData = np.array(msrData).T
      targData = np.array(targData).T
      outputArray = PCM(featData, msrData, targData)

    if pcmVersion=='snapshot':
      name = "time"
      outputDict[name] = np.asarray(time)
      name = "snapshot_pri_post_stdReduct"
      outputDict[name] = np.asarray(outputArray)
    if pcmVersion=='static':
      for targ in self.targets:
        name = "static_pri_post_stdReduct_" + targ.split('|')[-1]
        outputDict[name] = np.asarray(outputArray)
    if pcmVersion=='Tdep':
      name = "time"
      outputDict[name] = np.asarray(time)
      name = "Tdep_post_mean"
      outputDict[name] = np.asarray(yAppPred)
      name = "Error"
      outputDict[name] = np.asarray(error)

    return outputDict


