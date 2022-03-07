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
  MultinomialNBClassifier
  Naive Bayes classifier for multinomial models
"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class MultinomialNB(ScikitLearnBase):
  """
    MultinomialNBClassifier
    Naive Bayes classifier for multinomial models
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
    self.model = sklearn.naive_bayes.MultinomialNB

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(MultinomialNB, cls).getInputSpecification()
    specs.description = r"""The \\textit{MultinomialNB} classifier implements the naive Bayes algorithm for
                        multinomially distributed data, and is one of the two classic naive Bayes
                        variants used in text classification (where the data is typically represented
                        as word vector counts, although tf-idf vectors are also known to work well in
                        practice).
                        The distribution is parametrized by vectors $\theta_y =
                        (\theta_{y1},\ldots,\theta_{yn})$ for each class $y$, where $n$ is the number of
                        features (in text classification, the size of the vocabulary) and $\theta_{yi}$
                        is the probability $P(x_i \mid y)$ of feature $i$ appearing in a sample
                        belonging to class $y$.
                        The parameters $\theta_y$ are estimated by a smoothed version of maximum
                        likelihood, i.e. relative frequency counting:
                        \begin{equation}
                        \hat{\theta}_{yi} = \frac{ N_{yi} + \alpha}{N_y + \alpha n}
                        \end{equation}
                        where $N_{yi} = \sum_{x \in T} x_i$ is the number of times feature $i$ appears
                        in a sample of class y in the training set T, and
                        $N_{y} = \sum_{i=1}^{|T|} N_{yi}$ is the total count of all features for class
                        $y$.
                        The smoothing priors $\alpha \ge 0$ account for features not present in the
                        learning samples and prevents zero probabilities in further computations.
                        Setting $\alpha = 1$ is called Laplace smoothing, while $\alpha < 1$ is called
                        Lidstone smoothing.
                        \zNormalizationPerformed{MultinomialNB}
                        """
    specs.addSub(InputData.parameterInputFactory("class_prior", contentType=InputTypes.FloatListType,
                                                  descr=r"""Prior probabilities of the classes. If specified the priors are
                                                  not adjusted according to the data. \nb the number of elements inputted here must
                                                  match the number of classes in the data set used in the training stage.""", default=None))
    specs.addSub(InputData.parameterInputFactory("alpha", contentType=InputTypes.FloatType,
                                                 descr=r"""Additive (Laplace and Lidstone) smoothing parameter (0 for no smoothing).
                                                 """, default=1.0))
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
    settings, notFound = paramInput.findNodesAndExtractValues(['class_prior', 'alpha', 'alpha'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
