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
  GaussianNB Classifier
  Gaussian Naive Bayes (GaussianNB) classifier

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class GaussianNB(ScikitLearnBase):
  """
    GaussianNB Classifier
    Gaussian Naive Bayes (GaussianNB) classifier
  """
  info = {'problemtype':'classification', 'normalize':True}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    import sklearn
    import sklearn.naive_bayes
    self.model = sklearn.naive_bayes.GaussianNB

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(GaussianNB, cls).getInputSpecification()
    specs.description = r"""The \\textit{GaussianNB} classifier implements the Gaussian Naive Bayes
                         algorithm for classification.
                         The likelihood of the features is assumed to be Gaussian:
                         \begin{equation}
                             P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i -
                               \mu_y)^2}{2\sigma^2_y}\right)
                         \end{equation}
                         The parameters $\sigma_y$ and $\mu_y$ are estimated using maximum likelihood.
                         \zNormalizationPerformed{GaussianNB}
                         """
    specs.addSub(InputData.parameterInputFactory("priors", contentType=InputTypes.FloatListType,
                                                  descr=r"""Prior probabilities of the classes. If specified the priors are
                                                  not adjusted according to the data. \nb the number of elements inputted here must
                                                  match the number of classes in the data set used in the training stage.""", default=None))
    specs.addSub(InputData.parameterInputFactory("var_smoothing", contentType=InputTypes.FloatType,
                                                 descr=r"""Portion of the largest variance of all features that is added to variances for
                                                 calculation stability.""", default=1e-9))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['priors', 'var_smoothing'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
