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
  Created on Nov. 22, 2021

  @author: wangc
  StackingRegressor
  A Bagging regressor.
"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class StackingRegressor(ScikitLearnBase):
  """
    Stack of estimators with a final regressor.
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

    self.model = sklearn.ensemble.StackingRegressor

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
    specs.description = r"""The \xmlNode{StackingRegressor} consists in stacking the output of individual estimator and
                          use a regressor to compute the final prediction. Stacking allows to use the strength of each
                          individual estimator by using their output as input of a final estimator.
                         """
    estimatorInput = InputData.assemblyInputFactory("estimator", contentType=InputTypes.StringType,
                                                 descr=r"""name of a ROM that can be used as an estimator""", default='no-default')
    specs.addSub(estimatorInput)
    specs.addSub(InputData.parameterInputFactory("final_estimator", contentType=InputTypes.StringType,
                                                 descr=r"""The name of estimator which will be used to combine the base estimators.""", default='no-default'))
    specs.addSub(InputData.parameterInputFactory("cv", contentType=InputTypes.IntegerType,
                                                 descr=r"""specify the number of folds in a (Stratified) KFold,""", default=5))
    specs.addSub(InputData.parameterInputFactory("passthrough", contentType=InputTypes.BoolType,
                                                 descr=r"""When False, only the predictions of estimators will be used as training
                                                 data for final\_estimator. When True, the final\_estimator is trained on the predictions
                                                 as well as the original training data.""", default=False))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['final_estimator', 'cv', 'passthrough'])
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
    foundFinalEstimator = False
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
      if estimator.name == self.settings['final_estimator']:
        self.settings['final_estimator'] = sklEstimator
        foundFinalEstimator = True
        continue
      estimators.append((estimator.name, sklEstimator))
    self.settings['estimators'] = estimators
    if not foundFinalEstimator:
      self.raiseAnError(IOError, 'final_estimator:', self.settings['final_estimator'], 'is not found among provdide estimators:',
                        ','.join([name for name,_ in estimators]))
    self.initializeModel(self.settings)
