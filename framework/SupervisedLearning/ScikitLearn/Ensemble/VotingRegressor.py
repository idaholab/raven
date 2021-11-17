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
  Created on Jan 21, 2020

  @author: alfoa, wangc
  ExtraTreeRegressor
  An extremely randomized tree regressor.

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from SupervisedLearning.ScikitLearn import ScikitLearnBase
from utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class VotingRegressor(ScikitLearnBase):
  """
    Prediction voting regressor for unfitted estimators.

    A voting regressor is an ensemble meta-estimator that fits several base regressors, each on the whole dataset.
    Then it averages the individual predictions to form a final predictions.
  """
  info = {'problemtype':'regression', 'normalize':True}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.multioutputWrapper = False
    import sklearn
    import sklearn.ensemble
    self.model = sklearn.ensemble.VotingRegressor
    self.settings = None #

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
    specs.description = r"""The \xmlNode{VotingRegressor}
                         \zNormalizationPerformed{VotingRegressor}
                         """
    estimatorInput = InputData.assemblyInputFactory("estimator", contentType=InputTypes.StringType,
                                                 descr=r"""name of a ROM that can be used as an estimator""", default='no-default')
    #TODO: Add more inputspecs for estimator
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

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    params = self.settings
    return params

  def setEstimator(self, estimatorList):
    """
      Initialization method
      @ In, estimatorList, list of ROM instances/estimators used by ROM
      @ Out, None
    """
    estimators = []
    for estimator in estimatorList:
      interfaceRom = estimator._interfaceROM
      if not isinstance(interfaceRom, ScikitLearnBase):
        self.raiseAnError(IOError, 'ROM', estimator.name, 'can not be used as estimator for ROM', self.name)
      if not callable(getattr(interfaceRom.model, "fit", None)):
        self.raiseAnError(IOError, 'estimator:', estimator.name, 'can not be used! Please change to a different estimator')
      else:
        self.raiseADebug('A valid estimator', estimator.name, 'is provided!')
      estimators.append((estimator.name, interfaceRom.model))
    self.model = self.model(estimators)
    self.initializeModel(self.settings)
