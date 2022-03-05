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
from utils import utils, randomUtils
from utils import InputData, InputTypes
from .. import ValidationBase
#Internal Modules End--------------------------------------------------------------------------------

class PhysicsGuidedCoverageMapping(ValidationBase):
  """
    PCM metric class.
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
    measurementsInput.addParam("type", InputTypes.StringType)
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
    self.dynamicType = ['static','dynamic'] #  for now only static is available
    self.acceptableMetrics = ["CDFAreaDifference", "PDFCommonArea", "STDReduction"] #  acceptable metrics
    self.name = 'PhysicsGuidedCoverageMapping'

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName() == 'Measurements':
        self.measurements = child.value    


  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject
      @ In, inputIn, list, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    dataDict = {self.getDataSetName(data): data for _, _, data in inputIn['Data']}
    pivotParameter = self.pivotParameter
    names = [self.getDataSetName(inp[-1]) for inp in inputIn['Data']]
    if len(inputIn['Data'][0][-1].indexes) and self.pivotParameter is None:
      if 'dynamic' not in self.dynamicType:
        self.raiseAnError(IOError, "The validation algorithm '{}' is not a dynamic model but time-dependent data has been inputted in object {}".format(self._type, inputIn['Data'][0][-1].name))
    evaluation ={k: np.atleast_1d(val) for k, val in  self._evaluate(dataDict, **{'dataobjectNames': names}).items()}

    if pivotParameter:
      if len(inputIn['Data'][0][-1]['time']) != len(list(evaluation.values())[0]):
        self.raiseAnError(RuntimeError, "The pivotParameter value '{}' has size '{}' and validation output has size '{}'".format( len(dataSets[0][self.pivotParameter]), len(evaluation.values()[0])))
      if pivotParameter not in evaluation:
        evaluation[pivotParameter] = inputIn['Data'][0][-1]['time']
    return evaluation

  ### utility functions
  def _evaluate(self, datasets, **kwargs):
    """
      Main method to "do what you do".
      @ In, datasets, list, list of datasets (data1,data2,etc.) to used.
      @ In, kwargs, dict, keyword arguments
      @ Out, outputDict, dict, dictionary containing the results {"feat"_"target"_"metric_name":value}
    """
    names = kwargs.get('dataobjectNames')
    outputDict = {}
    for feat, targ, msr in zip(self.features, self.targets, self.measurements):
      featData = self._getDataFromDataDict(datasets, feat, names)
      targData = self._getDataFromDataDict(datasets, targ, names)
      msrData = self._getDataFromDataDict(datasets, msr, names)

    # Values from first entry of tuple data from _getDataFromDataDict
    yExp = np.array(featData[0])
    yApp = np.array(targData[0])
    yMsr = np.array(msrData[0])
    # Sample mean as reference value
    yExpRef = np.mean(yExp, axis=0)
    yAppRef = np.mean(yApp)
    # Standardization
    yExpStd = (yExp-yExpRef)/yExpRef
    yAppStd = (yApp-yAppRef)/yAppRef
    if yExp.shape[1]!= yMsr.shape[1]:
      self.raiseAnError(IOError, "Number of Measurements is not consistent with number of Experiments/Features.")
    yMsrStd = (yMsr-yExpRef)/yExpRef

    # Single Experiment response
    if yExpStd.shape[1]==1:
      yExpReg = yExpStd.flatten()
    # Pseudo response of multiple Experiment responses
    # OrthogonalMatchingPursuit from sklearn used here
    # Possibly change to other regressors      
    elif yExpStd.shape[1]>1:
      regrExp = OrthogonalMatchingPursuit(fit_intercept=False).fit(yExpStd, yAppStd)
      yExpReg = regrExp.predict(yExpStd)
    
    # Kernel Desnity Estimation
    m1 = yExpReg[:]
    m2 = yAppStd.flatten()
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()

    binKDE = 200j
    # Grid of Experiment (X), grid of Application (Y)
    X, Y = np.mgrid[xmin:xmax:binKDE, ymin:ymax:binKDE]
    psts = np.vstack([X.ravel(), Y.ravel()])
    vals = np.vstack([m1, m2])
    # kernel
    knl = stats.gaussian_kde(vals)
    # Joint PDF of Experiment and Application
    Z = np.reshape(knl(psts).T, X.shape)

    if yMsrStd.shape[1]==1:
      yMsrStd = yMsrStd.flatten()
    # Combine measurements by multiple Experiment regression
    elif yMsrStd.shape[1]>1:
      yMsrStd = regrExp.predict(yMsrStd)
    
    # Measurement PDF with KDE
    knlMsr = stats.gaussian_kde(yMsrStd)
    pdfMsr = knlMsr(X[:, 0])

    # yAppPred by integrating p(yexp, yapp)dyexp * p(yexp)
    # on range [ymsr.min(), ymsr.max()]
    pdfAppPred = np.zeros(Y.shape[1])
    intgrPdf = 0.0 # for normalization
    for i in range(len(Y[0, :])):
      pAppi = 0.0
      for j in range(len(X[:, 0])):
        pAppi += knl.evaluate([X[j, 0], Y[0, i]]) * np.diff(X[:, 0])[0] * pdfMsr[j]
      pdfAppPred[i] = pAppi
      intgrPdf += pdfAppPred[i]*(Y[0, 1]-Y[0, 0])

    # normalized PDF of predicted application
    pdfAppPredNorm = pdfAppPred/intgrPdf

    # Calculate Expectation (average value) of predicted application
    predMean = 0.0
    for i in range(len(Y[0, :])):
      predMean += Y[0, i]*pdfAppPredNorm[i]*(Y[0, 1]-Y[0, 0])

    # Calculate Variance of predicted application
    predVar = 0.0
    for i in range(len(Y[0, :])):
      predVar += (Y[0, i]-predMean)**2.0 * pdfAppPredNorm[i]*(Y[0, 1]-Y[0, 0])

    # Standard Deviation
    predStd = np.sqrt(predVar)
    priStd = np.std(yAppStd)
    stdReduct = 1.0-predStd/priStd

    for metric in self.metrics:
      name = "pri_post_{}".format(metric.estimator.name)
      outputDict[name] = stdReduct

    return outputDict
