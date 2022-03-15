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
  KNeighborsClassifier
  Classifier implementing the k-nearest neighbors vote.

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class KNeighborsClassifier(ScikitLearnBase):
  """
    KNeighborsClassifier
    Classifier implementing the k-nearest neighbors vote.
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
    import sklearn.neighbors
    self.model = sklearn.neighbors.KNeighborsClassifier

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(KNeighborsClassifier, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{KNeighborsClassifier} is a type of instance-based learning or
                         non-generalizing learning: it does not attempt to construct a general internal
                         model, but simply stores instances of the training data.
                         Classification is computed from a simple majority vote of the nearest neighbors
                         of each point: a query point is assigned the data class which has the most
                         representatives within the nearest neighbors of the point.
                         It implements learning based on the $k$ nearest neighbors of each query point,
                         where $k$ is an integer value specified by the user.
                         \zNormalizationPerformed{KNeighborsClassifier}
                        """
    # penalty
    specs.addSub(InputData.parameterInputFactory("n_neighbors", contentType=InputTypes.IntegerType,
                                                 descr=r"""Number of neighbors to use by default for kneighbors queries.""", default=5))
    specs.addSub(InputData.parameterInputFactory("weights", contentType=InputTypes.makeEnumType("weight", "weightType",['uniform','distance']),
                                                 descr=r"""weight function used in prediction. If ``uniform'', all points in each neighborhood
                                                 are weighted equally. If ``distance'' weight points by the inverse of their distance. in this
                                                 case,closer neighbors of a query point will have a greater influence than neighbors which are
                                                 further away. """, default='uniform'))
    specs.addSub(InputData.parameterInputFactory("algorithm", contentType=InputTypes.makeEnumType("algorithm", "algorithmType",['auto','ball_tree','kd_tree','brute']),
                                                 descr=r"""Algorithm used to compute the nearest neighbors """, default='auto'))
    specs.addSub(InputData.parameterInputFactory("leaf_size", contentType=InputTypes.IntegerType,
                                                 descr=r"""Leaf size passed to BallTree or KDTree. This can affect the speed of the construction
                                                 and query, as well as the memory required to store the tree. The optimal value depends on the
                                                 nature of the problem.""", default=30))
    specs.addSub(InputData.parameterInputFactory("p", contentType=InputTypes.IntegerType,
                                                 descr=r"""Power parameter for the Minkowski metric. When $p = 1$, this is equivalent to using
                                                 manhattan\_distance (l1), and euclidean\_distance (l2) for $p = 2$. For arbitrary $p$, minkowski\_distance
                                                 (l\_p) is used.""", default=2))
    specs.addSub(InputData.parameterInputFactory("metric", contentType=InputTypes.makeEnumType("metric", "metricType",['euclidean','manhattan', 'minkowski',
                                                 'chebyshev', 'hamming', 'braycurtis']),
                                                 descr=r"""the distance metric to use for the tree. The default metric is minkowski, and with
                                                 $p=2$ is equivalent to the standard Euclidean metric.
                                                 The available metrics are:
                                                 \begin{itemize}
                                                   \item minkowski: $sum(|x - y|^p)^(1/p)$
                                                   \item euclidean: $sqrt(sum((x - y)^2))$
                                                   \item manhattan: $sum(|x - y|)$
                                                   \item chebyshev: $max(|x - y|)$
                                                   \item hamming: $N\_unequal(x, y) / N\_tot$
                                                   \item canberra: $sum(|x - y| / (|x| + |y|))$
                                                   \item braycurtis: $sum(|x - y|) / (sum(|x|) + sum(|y|))$
                                                 \end{itemize}
                                                 """, default='minkowski'))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['n_neighbors', 'weights', 'algorithm',
                                                               'leaf_size', 'p','metric'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
