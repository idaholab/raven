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
  Created on Nov. 16, 2021

  @author: wangc
  VotingRegressor
  A voting regressor is an ensemble meta-estimator that fits several base regressors
"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class VotingRegressor(ScikitLearnBase):
  """
    Prediction voting regressor for unfitted estimators.

    A voting regressor is an ensemble meta-estimator that fits several base regressors, each on the whole dataset.
    Then it averages the individual predictions to form a final predictions.
  """
  info = {'problemtype':'regression', 'normalize':False}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.multioutputWrapper = True
    import sklearn
    import sklearn.ensemble
    self.model = sklearn.ensemble.VotingRegressor

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.description = r"""The \xmlNode{VotingRegressor} is an ensemble meta-estimator that fits several base
                            regressors, each on the whole dataset. Then it averages the individual predictions to form
                            a final prediction.
                         """
    estimatorInput = InputData.assemblyInputFactory("estimator", contentType=InputTypes.StringType,
                                                 descr=r"""name of a ROM that can be used as an estimator""", default='no-default')
    specs.addSub(estimatorInput)
    specs.addSub(InputData.parameterInputFactory("weights", contentType=InputTypes.FloatListType,
                                                 descr=r"""Sequence of weights (float or int) to weight the occurrences of predicted
                                                 values before averaging. Uses uniform weights if None.""", default=None))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['weights'])
    # notFound must be empty
    assert(not notFound)
    self.settings = settings

  def setEstimator(self, estimatorList):
    """
      Initialization method
      @ In, estimatorList, list of ROM instances/estimators used by ROM
      @ Out, None
    """
    super().setEstimator(estimatorList)
    estimators = []
    for estimator in estimatorList:
      interfaceRom = estimator._interfaceROM
      if interfaceRom.info['problemtype'] != 'regression':
        self.raiseAnError(IOError, 'estimator:', estimator.name, 'with problem type', interfaceRom.info['problemtype'],
                          'can not be used for', self.name)
      # In sklearn, multioutput wrapper can not be used by outer and inner estimator at the same time
      # If the outer estimator can handle multioutput, the multioutput wrapper of inner can be kept,
      # otherwise, we need to remove the wrapper for inner estimator.
      if interfaceRom.multioutputWrapper:
        sklEstimator = interfaceRom.model.get_params()['estimator']
      else:
        sklEstimator = interfaceRom.model
      estimators.append((estimator.name, sklEstimator))
    self.settings['estimators'] = estimators
    self.initializeModel(self.settings)
