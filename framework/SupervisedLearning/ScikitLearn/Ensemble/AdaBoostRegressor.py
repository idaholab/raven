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
  AdaBoostRegressor
  An AdaBoost regressors
"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class AdaBoostRegressor(ScikitLearnBase):
  """
    An AdaBoost regressors
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
    self.model = sklearn.ensemble.AdaBoostRegressor

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
    specs.description = r"""The \xmlNode{AdaBoostRegressor} is a meta-estimator that begins by fitting a regressor on
                            the original dataset and then fits additional copies of the regressor on the same dataset
                            but where the weights of instances are adjusted according to the error of the current
                            prediction. As such, subsequent regressors focus more on difficult cases.
                         """
    estimatorInput = InputData.assemblyInputFactory("estimator", contentType=InputTypes.StringType,
                                                 descr=r"""name of a ROM that can be used as an estimator""", default='no-default')
    specs.addSub(estimatorInput)
    specs.addSub(InputData.parameterInputFactory("n_estimators", contentType=InputTypes.IntegerType,
                                                 descr=r"""The maximum number of estimators at which boosting is
                                                 terminated. In case of perfect fit, the learning procedure is
                                                 stopped early.""", default=50))
    specs.addSub(InputData.parameterInputFactory("learning_rate", contentType=InputTypes.FloatType,
                                                 descr=r"""Weight applied to each regressor at each boosting iteration.
                                                 A higher learning rate increases the contribution of each regressor.
                                                 There is a trade-off between the learning\_rate and n\_estimators
                                                 parameters.""", default=1.0))
    specs.addSub(InputData.parameterInputFactory("loss", contentType=InputTypes.makeEnumType("loss", "lossType",['linear', 'square', 'exponential']),
                                                 descr=r"""The loss function to use when updating the weights after each
                                                 boosting iteration.""", default='linear'))
    specs.addSub(InputData.parameterInputFactory("random_state", contentType=InputTypes.IntegerType,
                                                 descr=r"""Controls the random seed given at each estimator at each
                                                 boosting iteration.""", default=None))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['n_estimators', 'learning_rate', 'loss', 'random_state'])
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
    if len(estimatorList) != 1:
      self.raiseAWarning('ROM', self.name, 'can only accept one estimator, but multiple estimators are provided!',
                          'Only the first one will be used, i.e.,', estimator.name)
    estimator = estimatorList[0]
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
    settings = {'base_estimator':sklEstimator}
    self.settings.update(settings)
    self.initializeModel(self.settings)
