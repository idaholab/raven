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
  BaggingRegressor
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

class BaggingRegressor(ScikitLearnBase):
  """
    A Bagging Regressor
    A Bagging regressor is an ensemble meta-estimator that fits base regressors each on random subsets of the original
    dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final
    prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator
    (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble
    out of it.

    This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as
    random subsets of the samples, then this algorithm is known as Pasting. If samples are drawn with replacement,
    then the method is known as Bagging. When random subsets of the dataset are drawn as random subsets of the
    features, then the method is known as Random Subspaces. Finally, when base estimators are built on subsets of
    both samples and features, then the method is known as Random Patches.
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
    self.model = sklearn.ensemble.BaggingRegressor

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
    specs.description = r"""The \xmlNode{BaggingRegressor} is an ensemble meta-estimator that fits base regressors each on random subsets of the original
                            dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final
                            prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator
                            (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble
                            out of it.
                         """
    estimatorInput = InputData.assemblyInputFactory("estimator", contentType=InputTypes.StringType,
                                                 descr=r"""name of a ROM that can be used as an estimator""", default='no-default')
    specs.addSub(estimatorInput)
    specs.addSub(InputData.parameterInputFactory("n_estimators", contentType=InputTypes.IntegerType,
                                                 descr=r"""The number of base estimators in the ensemble.""", default=10))
    specs.addSub(InputData.parameterInputFactory("max_samples", contentType=InputTypes.FloatType,
                                                 descr=r"""The number of samples to draw from X to train each base estimator""", default=1.0))
    specs.addSub(InputData.parameterInputFactory("max_features", contentType=InputTypes.FloatType,
                                                 descr=r"""The number of features to draw from X to train each base estimator """, default=1.0))
    specs.addSub(InputData.parameterInputFactory("bootstrap", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether samples are drawn with replacement. If False, sampling without
                                                 replacement is performed.""", default=True))
    specs.addSub(InputData.parameterInputFactory("bootstrap_features", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether features are drawn with replacement.""", default=False))
    specs.addSub(InputData.parameterInputFactory("oob_score", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether to use out-of-bag samples to estimate the generalization error.
                                                 Only available if bootstrap=True.""", default=False))
    specs.addSub(InputData.parameterInputFactory("warm_start", contentType=InputTypes.BoolType,
                                                 descr=r"""When set to True, reuse the solution of the previous call to fit and add more
                                                 estimators to the ensemble, otherwise, just fit a whole new ensemble.""", default=False))
    specs.addSub(InputData.parameterInputFactory("random_state", contentType=InputTypes.IntegerType,
                                                 descr=r"""Controls the random resampling of the original dataset (sample wise and feature wise). """,
                                                 default=None))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['n_estimators', 'max_samples', 'max_features', 'bootstrap', 'bootstrap_features',
                                    'oob_score', 'warm_start', 'random_state'])
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
