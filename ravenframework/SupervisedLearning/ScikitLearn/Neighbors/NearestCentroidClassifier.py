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
  Nearest Centroid Classifier

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....SupervisedLearning.ScikitLearn import ScikitLearnBase
from ....utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class NearestCentroid(ScikitLearnBase):
  """
    Nearest Centroid Classifier
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
    self.model = sklearn.neighbors.NearestCentroid

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
    specs.description = r"""The \xmlNode{RadiusNeighborsClassifier} is a type of instance-based learning or
                         non-generalizing learning: it does not attempt to construct a general internal
                         model, but simply stores instances of the training data.
                         Classification is computed from a simple majority vote of the nearest neighbors
                         of each point: a query point is assigned the data class which has the most
                         representatives within the nearest neighbors of the point.
                         It implements learning based on the number of neighbors within a fixed radius
                         $r$ of each training point, where $r$ is a floating-point value specified by the
                         user.
                         \zNormalizationPerformed{RadiusNeighborsClassifier}
                        """
    # penalty
    specs.addSub(InputData.parameterInputFactory("shrink_threshold", contentType=InputTypes.FloatType,
                                                 descr=r"""Threshold for shrinking centroids to remove features.""", default=None))
    specs.addSub(InputData.parameterInputFactory("metric", contentType=InputTypes.makeEnumType("metric", "metricType",['uniform','distance']),
                                                 descr=r"""The metric to use when calculating distance between instances in a feature array.
                                                 The available metrics are allo the ones explained in the \xmlNode{Metrics} section (pairwise).
                                                 The centroids for the samples corresponding to each class is the point from which the sum of
                                                 the distances (according to the metric) of all samples that belong to that particular class are
                                                 minimized. If the ``manhattan'' metric is provided, this centroid is the median and for all other metrics,
                                                 the centroid is now set to be the mean.
                                                 """, default='minkowski'))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['shrink_threshold', 'metric'])
    # notFound must be empty
    assert(not notFound)
    self.initializeModel(settings)
