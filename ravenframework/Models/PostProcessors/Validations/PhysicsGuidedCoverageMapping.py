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
    pcmTypeInput = InputData.parameterInputFactory("pcmType", contentType=InputTypes.StringType)
    specs.addSub(pcmTypeInput)
    recErrorInput = InputData.parameterInputFactory("ReconstructionError", contentType=InputTypes.FloatType)
    specs.addSub(recErrorInput)
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
    for child in paramInput.subparts:
      # Get option for type of PCM to be run from input
      if child.getName() == 'pcmType':
        self.pcmType = child.value
      # Get Measurements data from input
      elif child.getName() == 'Measurements':
        self.measurements = child.value
      # Get reconstruction error from input to decide on the rank in TdepPCM
      elif child.getName() == 'ReconstructionError':
        self.ReconstructionError = child.value
      else :
        self.ReconstructionError = 0.001
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
    evaluation ={k: np.atleast_1d(val) for k, val in self._evaluate(dataDict, **{'dataobjectNames': names}).items()}
    if 'time' in evaluation.keys():
      pcmType = self.pcmType
      evaluation[pcmType] = inputIn['Data'][0][-1]['time']
    return evaluation

  def _evaluate(self, datasets, **kwargs):
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
    # Find a reference sample which is closest to samples' average for scaling later
    def FindRef(yExp, yApp):
      """
      Method to find reference samples within experimental and application data sets that are
      the closest to the respective sample mean. The reference samples are determined by
      minimizing the mean squared error between each sample and the sample mean.

      @ In, yExp, yApp: array-like (2D), experiment and application data [samples x timesteps].
      @ Out, inExp, inApp: int, indexes of the reference sample in the experimental/application dataset.
      """
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


    def PCM(featData, msrData, targData):
      """
      This is the PCM function for Static PCM.
      Snapshot PCM is also applying this function sequentially on data from each single timestep.

      It takes three input matrices and compute the uncertainty reduction fraction
      for a given set of feature data, measurement data, and target data using Static PCM (OMP+KDE).
      The data is scaled first, then OMP finds pseudo responses 'yExpReg' based on training(featdata, targdata).
      Then pseudo response 'yMsrReg' is also obtained for slicing step in KDE later.
      By slicing at 'yMsrReg' for joint distribution cloud (yExpReg, yAppStd),
      we have posterior distribution for yApp, and its uncertainty reduction fraction is computed.

      @ In, featData, msrData, targData: array-like, should be formatted as [samples x timesteps].
      @ Out, outputArray: list, list of uncertainty reduction fractions for each target model/response.
      """
      outputArray = []

      yExp = np.array(featData)
      yMsr = np.array(msrData)
      # Reference values of Experiments, yExpRef is one of the M samples
      # Which is the closest to sample mean by mean square errors
      # It can be user-defined in the future
      inExp, inApp = FindRef(featData, targData)
      # Reference values of Experiment, yExpRef is a scalar
      yExpRef = featData[inExp, :]
      # Reference values of Application, yAppRef is a scalar
      yAppRef = targData[inApp, :]

      for t in range(targData.shape[1]):
        # Application responses yApp in Nx1
        yApp = np.array(targData[:,t])
        # Usually the reference value is given,
        # and will not be zero, e.g. reference fuel temperature.
        # Standardization
        yExpStd = (np.delete(yExp[:,:] - yExpRef, inExp,0))
        yMsrStd = (yMsr-yExpRef)/yExpRef
        yAppStd = (np.delete(yApp - yAppRef[t], inApp,0))

        # For each Target/Application model/response, calculate an uncertainty reduction fraction
        # using all available Features/Experiments
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

    # Functions below, KNN, FindRank, pcmTdep, are for T-Dep PCM
    def KNN(exp, app, expM, k=5, Metric='minkowski'):
      """
      Method to perform K-Nearest Neighbors (KNN) regression on a given set of experimental data
      and application data, and make predictions on application measurement using the fitted model.

      @ In, exp: array-like (2D), the experimental data used for training the KNN model.
      @ In, app: array-like (1D or 2D), the application data corresponding to the experimental data.
      @ In, expM: array-like (1D or 2D), the measurement data on which predictions are to be made.
      @ In, k: int, optional, the number of neighbors to be considered in KNN algorithm. Default is 5.
      @ In, Metric: str, optional, the distance metric to use for the KNN algorithm. Default is 'minkowski'.
      @ Out, appPred: array-like, the predicted application data for the given measurement set.
      """
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

    def FindRank(ref, x, recError):
      """
      Method to determine the rank of matrix 'x' such that the maximum residual for
      reconstructed matrix 'x_r' under different ranks is less than a error defined,
      i.e., max|x-x_r|<error. The error is decided based on measurement error given.
      Here we decide the error by a fraction of measurement standard deviation.

      @ In, ref: array-like (1D), reference sample used for scaling data 'x'.
      @ In, x: array-like (2D), the matrix [samples x timesteps] for which the rank is to be determined.
      @ In, recError, float, reconstruction error to determine the rank of time series data.
      @ Out, xMAX[0]: rank, int, the rank of matrix 'x' that satisfies the condition on maximum residual.
      """

      x = (x.T/ref).T

      #scaling by reference
      u, s, v_T = np.linalg.svd(x)
      # Find the maximum of the residual under different rank
      maxError = np.zeros(x.shape[0])
      for k in range(1,maxError.shape[0]+1):
        res = x - u[:,:k]@np.diag(s)[:k,:k]@v_T[:k,:]#residual of x-x_r under different rank
        maxError[k-1] = np.max(abs(res)) # maximum of the residual

      # The precition of pcmTdep is sensitive to the rank here, which determines the dimension of
      # U_exp, U_app subspaces for KNN regression. We will lose information under low rank, or
      # overfit data with noise under high rank.
      # This error can be made as user-defined.
      # 0.001 is decided as we want a rank corresponding to around 0.1% reconstruction error.
      xMAX=min(np.argwhere(maxError<recError)) #rank corresponding to reconstruction error
      return xMAX[0]

    def pcmTdep(featData, msrData, targData, recError):
      """
        Method to perform Tdep PCM model for uncertainty quantification and data assimilation in time-dependent
        applications using Singular Value Decomposition (SVD) and K-Nearest Neighbors regression(KNN).
        The inputs, featData, msrData, targData, are scaled by reference first, then applying SVD
        to find U_exp, U_app subspace, coefficients under relevant U subspace, and decide on the rank.
        The coefficients found are fitted with KNN, then predicted application response is reconstructed
        by the estimated application coefficients.

        @ In, featData, msrData,, targData : array-like (2D), the input data for features, measurements, and targets.
        @ In, recError, float, reconstruction error to determine the rank of time series data.
        @ Out, yAppPred: array-like (2D), the predicted application response.
        @ Out, error, array-like (2D), the relative error between predicted and computed application reference.
      """
      yExp = np.array(featData)
      yApp = np.array(targData)
      yExpMsr = np.array(msrData)
      yExpMsrMean = np.mean(np.array(msrData), axis=0)

      # calculate alphaApp as linear combination of alphaExp (with RankApp, RankExp)
      inExp, inApp = FindRef(yExp, yApp)
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
      rkApp = FindRank(yAppRef, yAppScaled, recError) #rank of application
      rkExp = FindRank(yExpRef, yExpScaled, recError)  # rank of experiments

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

    featData = np.array(featData)
    msrData = np.array(msrData)
    targData = np.array(targData)
    featPW = np.array(featPW).T
    msrPW = np.array(msrPW).T
    targPW = np.array(targPW).T

    pcmVersion = self.pcmType
    recError = self.ReconstructionError #reconstruction error to determine the rank of time series data.


    if pcmVersion == 'Tdep':
      self.raiseAMessage('***    Running Tdep-PCM       ***')
      # Data of size (num_of_samples, num_of_features)
      element = np.asarray(datasets['exp'].get('time'))[0]
      v = np.asarray(datasets['exp'].get('time'))
      num_of_samples = np.count_nonzero(v == element)
      num_of_featuresExp = int(np.asarray(datasets['exp'].get('time')).shape[0]/num_of_samples)
      num_of_featuresApp = int(np.asarray(datasets['app'].get('time')).shape[0]/num_of_samples)

      if featData.size != num_of_samples * num_of_featuresExp or \
        targData.size != num_of_samples * num_of_featuresApp:
        self.raiseAnError(IOError, 'The data provided in XML node "Features/Target" is not following the dimension required by TdepPCM')

      featData = np.array(featData).reshape(num_of_samples, num_of_featuresExp)
      msrData = np.array(msrData).reshape(num_of_samples, num_of_featuresExp)
      targData = np.array(targData).reshape(num_of_samples, num_of_featuresApp)
      time = v[:num_of_featuresExp]
      yAppPred, error = pcmTdep(featData, msrData, targData, recError)
      name = "time"
      outputDict[name] = np.asarray(time)
      name = "Tdep_post_mean"
      outputDict[name] = np.asarray(yAppPred)
      name = "Error"
      outputDict[name] = np.asarray(error)

    if pcmVersion == 'Snapshot':
      self.raiseAMessage('***    Running Snapshot-PCM   ***')
      # Data of size (num_of_samples, num_of_features)
      element = np.asarray(datasets['exp'].get('time'))[0]
      v = np.asarray(datasets['exp'].get('time'))
      num_of_samples = np.count_nonzero(v == element)
      num_of_features = int(np.asarray(datasets['exp'].get('time')).shape[0]/num_of_samples)
      num_of_targets = int(np.asarray(datasets['app'].get('time')).shape[0]/num_of_samples)

      if featData.size != num_of_samples * num_of_features or \
        targData.size != num_of_samples * num_of_targets or \
        num_of_features !=  num_of_targets:
        self.raiseAnError(IOError, 'The data provided in XML node "Features/Target" is not following the dimension required by SnapshotPCM')

      featData = np.array(featData).reshape(num_of_samples, num_of_features)
      msrData = np.array(msrData).reshape(num_of_samples, num_of_features)
      targData = np.array(targData).reshape(num_of_samples, num_of_targets)
      time = v[:num_of_features]
      outputArray = PCM(featData, msrData, targData)
      name = "time"
      outputDict[name] = np.asarray(time)
      name = "snapshot_pri_post_stdReduct"
      outputDict[name] = np.asarray(outputArray)

    if pcmVersion == 'Static':
      self.raiseAMessage('***     Running Static-PCM    ***')
      featData = np.array(featData).T
      msrData = np.array(msrData).T
      targData = np.array(targData).T
      outputArray = PCM(featData, msrData, targData)
      for targ in self.targets:
        name = "static_pri_post_stdReduct_" + targ.split('|')[-1]
        outputDict[name] = np.asarray(outputArray)

    return outputDict


