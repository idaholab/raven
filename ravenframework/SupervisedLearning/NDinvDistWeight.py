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
  Created on May 8, 2018

  @author: mandd, talbpaul, wangc
  Originally from SupervisedLearning.py, split in PR #650 in July 2018
  Specific ROM implementation for NDinvDistWeight ROM
"""

#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..utils import utils
from ..utils import InputData, InputTypes
interpolationND = utils.findCrowModule("interpolationND")
from .NDinterpolatorRom import NDinterpolatorRom
#Internal Modules End--------------------------------------------------------------------------------

class NDinvDistWeight(NDinterpolatorRom):
  """
    An N-dimensional model that interpolates data based on a inverse weighting of
    their training data points?
  """
  info = {'problemtype':'regression', 'normalize':True}
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super().getInputSpecification()
    specs.description = r"""The \xmlNode{NDinvDistWeight} is based on an
                            $N$-dimensional inverse distance weighting formulation.
                            Inverse distance weighting (IDW) is a type of deterministic method for
                            multivariate interpolation with a known scattered set of points.
                            The assigned values to unknown points are calculated via a weighted average of
                            the values available at the known points.
                            \\
                            \zNormalizationPerformed{NDinvDistWeight}
                            \\
                            In order to use this Reduced Order Model, the \xmlNode{ROM} attribute
                            \xmlAttr{subType} needs to be \xmlString{NDinvDistWeight}.
                        """
    specs.addSub(InputData.parameterInputFactory("p", contentType=InputTypes.IntegerType,
                                                 descr=r"""must be greater than zero and represents the ``power parameter''.
                                                 For the choice of value for \xmlNode{p},it is necessary to consider the degree
                                                 of smoothing desired in the interpolation/extrapolation, the density and
                                                 distribution of samples being interpolated, and the maximum distance over
                                                 which an individual sample is allowed to influence the surrounding ones (lower
                                                 $p$ means greater importance for points far away).""", default='no-default'))
    return specs

  def __init__(self):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'ND-INVERSEWEIGHT ROM'
    self._p = None # must be positive, power parameter

  def setInterpolator(self):
    """
      Set up the interpolator
      @ In, None
      @ Out, None
    """
    self.interpolator = []
    for _ in range(len(self.target)):
      self.interpolator.append(interpolationND.InverseDistanceWeighting(float(self._p)))

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    nodes, notFound = paramInput.findNodesAndExtractValues(['p'])
    if len(notFound) != 0:
      self.raiseAnError(IOError,'the <p> parameter must be provided in order to use NDinvDistWeigth as ROM!!!!')
    self._p = nodes['p']
    self.setInterpolator()

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    self.setInterpolator()
