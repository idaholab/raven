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
  Ridge Classifier with cross-validation

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class RidgeClassifierCV(ScikitLearnBase):
  """
    Ridge Classifier with cross-validation
  """
  info = {'problemtype':'regression', 'normalize':False}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    import sklearn
    import sklearn.linear_model
    self.model = sklearn.linear_model.RidgeClassifierCV

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(RidgeClassifierCV, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{RidgeClassifierCV} is a classifier that uses Ridge regression.
                        This classifier first converts the target values into {-1, 1} and then treats
                        the problem as a regression task (multi-output regression in the multiclass case).
                        In addition, a cross-validation method is applied to optimize the hyper-parameter.
                        By default, it performs Leave-One-Out Cross-Validation.
                        \zNormalizationNotPerformed{RidgeClassifierCV}
                        """
    specs.addSub(InputData.parameterInputFactory("fit_intercept", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether the intercept should be estimated or not. If False,
                                                  the data is assumed to be already centered.""", default=True))
    specs.addSub(InputData.parameterInputFactory("normalize", contentType=InputTypes.BoolType,
                                                 descr=r"""This parameter is ignored when fit_intercept is set to False. If True, the
                                                 regressors X will be normalized before regression by subtracting the mean and dividing
                                                 by the l2-norm. """, default=False))
    specs.addSub(InputData.parameterInputFactory("cv", contentType=InputTypes.IntegerType,
                                                 descr=r"""Determines the cross-validation splitting strategy.
                                                 It specifies the number of folds..""", default=None))
    specs.addSub(InputData.parameterInputFactory("alphas", contentType=InputTypes.FloatListType,
                                                 descr=r"""Array of alpha values to try. Regularization strength; must be a positive float. Regularization
                                                 improves the conditioning of the problem and reduces the variance of the estimates.
                                                 Larger values specify stronger regularization. Alpha corresponds to $1 / (2C)$ in other
                                                 linear models such as LogisticRegression or LinearSVC.""", default=[0.1, 1.0, 10.0]))
    specs.addSub(InputData.parameterInputFactory("scoring", contentType=InputTypes.StringType,
                                                 descr=r"""A string (see model evaluation documentation) or a scorer
                                                 callable object / function with signature.""", default=None))
    specs.addSub(InputData.parameterInputFactory("class_weight", contentType=InputTypes.makeEnumType("classWeight", "classWeightType",['balanced']),
                                                 descr=r"""If not given, all classes are supposed to have weight one.
                                                 The “balanced” mode uses the values of y to automatically adjust weights
                                                 inversely proportional to class frequencies in the input data""", default=None))
    specs.addSub(InputData.parameterInputFactory("store_cv_values", contentType=InputTypes.BoolType,
                                                 descr=r"""Flag indicating if the cross-validation values corresponding
                                                 to each alpha should be stored in the cv_values_ attribute (see below).
                                                 This flag is only compatible with cv=None (i.e. using Leave-One-Out
                                                 Cross-Validation).""", default=False))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['normalize','fit_intercept','cv', 'alphas',
                                                               'scoring', 'class_weight', 'store_cv_values'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
