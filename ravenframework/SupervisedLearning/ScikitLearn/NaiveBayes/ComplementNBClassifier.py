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
  ComplementNBClassifier
  Complement Naive Bayes classifier described in Rennie et al. (2003).

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class ComplementNB(ScikitLearnBase):
  """
    ComplementNB Classifier
    Complement Naive Bayes classifier described in Rennie et al. (2003).
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
    self.model = sklearn.naive_bayes.ComplementNB

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(ComplementNB, cls).getInputSpecification()
    specs.description = r"""The \\textit{ComplementNB} classifier (Complement Naive Bayes classifier) was designed to correct
                         the ``severe assumptions'' made by the standard Multinomial Naive Bayes classifier.
                         It is particularly suited for imbalanced data sets (see Rennie et al. (2003))
                         \zNormalizationPerformed{ComplementNB}
                         """
    specs.addSub(InputData.parameterInputFactory("alpha", contentType=InputTypes.FloatType,
                                                 descr=r"""Additive (Laplace and Lidstone) smoothing parameter (0 for no smoothing).
                                                 """, default=1.0))
    specs.addSub(InputData.parameterInputFactory("norm", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether or not a second normalization of the weights is performed.""", default=False))
    specs.addSub(InputData.parameterInputFactory("class_prior", contentType=InputTypes.FloatListType,
                                                  descr=r"""Prior probabilities of the classes. If specified the priors are
                                                  not adjusted according to the data. \nb the number of elements inputted here must
                                                  match the number of classes in the data set used in the training stage.""", default=None))
    specs.addSub(InputData.parameterInputFactory("fit_prior", contentType=InputTypes.BoolType,
                                                 descr=r"""Whether to learn class prior probabilities or not. If false, a uniform
                                                 prior will be used.""", default=True))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['alpha', 'norm','class_prior','fit_prior'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
