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
  Created on May 8, 2021

  @author: Andrea Alfonsi
  Base class for feature selection algorithm
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import abc
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ...utils import mathUtils
from ...utils import InputData, InputTypes
from ...BaseClasses import BaseInterface
#Internal Modules End--------------------------------------------------------------------------------

class FeatureSelectionBase(BaseInterface):
  """
    Feature selection base class
  """

  needROM = False

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    spec = super().getInputSpecification()
    spec.addSub(InputData.parameterInputFactory('parametersToInclude',
        contentType=InputTypes.StringListType,
        descr=r"""List of IDs of features/variables to include in the search.""", default=None))

    whichSpaceType = InputTypes.makeEnumType("spaceType","spaceTypeType",['Feature','feature','Target','target'])
    spec.addSub(InputData.parameterInputFactory('whichSpace',contentType=whichSpaceType,
        descr=r"""Which space to search? Target or Feature (this is temporary till """
        """DataSet training is implemented)""", default="feature"))
    return spec

  def __init__(self):
    super().__init__()
    self.parametersToInclude = None
    self.whichSpace =  None

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    nodes, notFound = paramInput.findNodesAndExtractValues(['parametersToInclude', 'whichSpace'])
    assert(not notFound)
    self.parametersToInclude = nodes['parametersToInclude']
    self.whichSpace = nodes['whichSpace'].lower()
    if self.parametersToInclude is None:
      self.raiseAnError(ValueError, '"parametersToInclude" must be inputted!' )

  def run(self, features, targets, X, y):
    """
      Run the feature selection model and then the underlying estimator
      on the selected features.
      @ In, features, list, list of features
      @ In, targets, list, list of targets
      @ In, X, numpy.array, feature data (nsamples,nfeatures) or (nsamples, nTimeSteps, nfeatures)
      @ In, y, numpy.array, target data (nsamples,nTargets) or (nsamples, nTimeSteps, nTargets)
      @ Out, newFeatures or newTargets, list, list of new features/targets
      @ Out, supportOfSupport_, np.array, boolean mask of the selected features
    """
    maskFeatures = None
    maskTargets = None
    #if self.parametersToInclude is not None:
    if self.whichSpace == 'feature':
      maskFeatures = [False]*len(features)
    else:
      maskTargets = [False]*len(targets)
    for param in self.parametersToInclude:
      if maskFeatures is not None and param in features:
        maskFeatures[features.index(param)] = True
      if maskTargets is not None and param in targets:
        maskTargets[targets.index(param)] = True
    if maskTargets is not None and np.sum(maskTargets) != len(self.parametersToInclude):
      self.raiseAnError(ValueError, "parametersToInclude are found in both feature and target spaces. Only one space is allowed!")
    if maskFeatures is not None and np.sum(maskFeatures) != len(self.parametersToInclude):
      self.raiseAnError(ValueError, "parametersToInclude are found in both feature and target spaces. Only one space is allowed!")
    return self._train(X, y, features, targets, maskF=maskFeatures, maskT=maskTargets)

  @abc.abstractmethod
  def _train(self, X, y, featuresIds, targetsIds, maskF = None, maskT = None):
    """
      Train the feature selection model and perform search of best features
      @ In, X, numpy.array, feature data (nsamples,nfeatures) or (nsamples, nTimeSteps, nfeatures)
      @ In, y, numpy.array, target data (nsamples,nTargets) or (nsamples, nTimeSteps, nTargets)
      @ In, featuresIds, list, list of features
      @ In, targetsIds, list, list of targets
      @ In, maskF, optional, np.array, indeces of features to search within
                                       (parameters to include None if search is whitin targets)
      @ In, maskT, optional, np.array, indeces of targets to search within
                                       (parameters to include None if search is whitin features)

      @ Out, newFeatures or newTargets, list, list of new features/targets
      @ Out, supportOfSupport_, np.array, boolean mask of the selected features
    """
    pass

  def var(self, variable):
    """
      Method to get an internal parameter
      @ In, variable, str, parameter that can be found in __dict__
      @ Out, var, instance, the instance of the requested variable (if not found, return None)
    """
    return self.__dict__.get(variable)
