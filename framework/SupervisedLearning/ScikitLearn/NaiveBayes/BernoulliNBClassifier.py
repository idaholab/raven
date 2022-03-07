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
  BernoulliNB Classifier
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

class BernoulliNB(ScikitLearnBase):
  """
    BernoulliNB Classifier
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
    self.model = sklearn.naive_bayes.BernoulliNB

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(BernoulliNB, cls).getInputSpecification()
    specs.description = r"""The \textit{BernoulliNB} classifier implements the naive Bayes training and
                         classification algorithms for data that is distributed according to multivariate
                         Bernoulli distributions; i.e., there may be multiple features but each one is
                         assumed to be a binary-valued (Bernoulli, boolean) variable.
                         Therefore, this class requires samples to be represented as binary-valued
                         feature vectors; if handed any other kind of data, a \textit{Bernoulli Naive
                         Bayes} instance may binarize its input (depending on the binarize parameter).
                         The decision rule for Bernoulli naive Bayes is based on
                         \begin{equation}
                         P(x_i \mid y) = P(i \mid y) x_i + (1 - P(i \mid y)) (1 - x_i)
                         \end{equation}
                         which differs from multinomial NB's rule in that it explicitly penalizes the
                         non-occurrence of a feature $i$ that is an indicator for class $y$, where the
                         multinomial variant would simply ignore a non-occurring feature.
                         In the case of text classification, word occurrence vectors (rather than word
                         count vectors) may be used to train and use this classifier.
                         \textit{Bernoulli Naive Bayes} might perform better on some datasets, especially
                         those with shorter documents.
                         \zNormalizationPerformed{BernoulliNB}
                         """
    specs.addSub(InputData.parameterInputFactory("alpha", contentType=InputTypes.FloatType,
                                                 descr=r"""Additive (Laplace and Lidstone) smoothing parameter (0 for no smoothing).
                                                 """, default=1.0))
    specs.addSub(InputData.parameterInputFactory("binarize", contentType=InputTypes.FloatType,
                                                 descr=r"""Threshold for binarizing (mapping to booleans) of sample features. If None,
                                                 input is presumed to already consist of binary vectors.
                                                 """, default=None))
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
    settings, notFound = paramInput.findNodesAndExtractValues(['alpha', 'binarize','class_prior','fit_prior'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
