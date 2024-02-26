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
Created on June 20, 2023
@author: j-bryan

Base classes for wrapping scikit-learn style transformers and characterizers.
"""

import abc
from copy import deepcopy

from ..TimeSeriesAnalyzer import TimeSeriesTransformer, TimeSeriesCharacterizer
from ...utils import xmlUtils, InputTypes


class SKLTransformer(TimeSeriesTransformer):
  """ Wrapper for scikit-learn transformers """
  _acceptsMissingValues = True

  @property
  @abc.abstractmethod
  def templateTransformer(self):
    """ Template transformer that must be implemented in child classes """

  def fit(self, signal, pivot, targets, settings):
    """
      Fits the algorithm/model using the provided time series ("signal") using methods specific to
      the algorithm.
      @ In, signal, np.array, time-dependent series
      @ In, pivot, np.array, time-like parameter
      @ In, targets, list(str), names of targets
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, params, dict, characterization of signal; structure as:
                           params[target variable][characteristic] = value
    """
    params = {}
    for tg, target in enumerate(targets):
      transformer = deepcopy(self.templateTransformer)
      transformer.fit(signal[:, tg].reshape(-1, 1))
      params[target] = {'model': transformer}
    return params

  def getResidual(self, initial, params, pivot, settings):
    """
      Removes trained signal from data and find residual
      @ In, initial, np.array, original signal shaped [pivotValues, targets], targets MUST be in
                               same order as self.target
      @ In, params, dict, training parameters as from self.characterize
      @ In, pivot, np.array, time-like array values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, residual, np.array, reduced signal shaped [pivotValues, targets]
    """
    residual = initial.copy()
    for tg, (target, data) in enumerate(params.items()):
      residual[:, tg] = data['model'].transform(residual[:, tg].reshape(-1, 1)).flatten()
    return residual

  def getComposite(self, initial, params, pivot, settings):
    """
      Combines two component signals to form a composite signal. This is essentially the inverse
      operation of the getResidual method.
      @ In, initial, np.array, original signal shaped [pivotValues, targets], targets MUST be in
                               same order as self.target
      @ In, params, dict, training parameters as from self.characterize
      @ In, pivot, np.array, time-like array values
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, composite, np.array, resulting composite signal
    """
    composite = initial.copy()
    for tg, (target, data) in enumerate(params.items()):
      composite[:, tg] = data['model'].inverse_transform(composite[:, tg].reshape(-1, 1)).flatten()
    return composite

  def _invertTransformationFunctions(self):
    """
      Swaps the forward and inverse functions of the template transformer.
      @ In, None
      @ Out, None
    """
    self.templateTransformer.transform, self.templateTransformer.inverse_transform = \
      self.templateTransformer.inverse_transform, self.templateTransformer.transform

  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, trained parameters as from self.fit
      @ Out, None
    """
    # Add model settings as subnodes to writeTO node
    for target, info in params.items():
      base = xmlUtils.newNode(target)
      writeTo.append(base)


class SKLCharacterizer(SKLTransformer, TimeSeriesCharacterizer):
  """ Wrapper for scikit-learn transformers that also provide a characterization of the data """
  def fit(self, signal, pivot, targets, settings):
    """
      Fits the algorithm/model using the provided time series ("signal") using methods specific to
      the algorithm.
      @ In, signal, np.array, time-dependent series
      @ In, pivot, np.array, time-like parameter
      @ In, targets, list(str), names of targets
      @ In, settings, dict, additional settings specific to algorithm
      @ Out, params, dict, characterization of signal; structure as:
                           params[target variable][characteristic] = value
    """
    params = {}
    for tg, target in enumerate(targets):
      transformer = deepcopy(self.templateTransformer)
      transformer.fit(signal[:, tg].reshape(-1, 1))
      # Attributes of interest in the transformer have the convention of ending with an underscore,
      # so that underscore is added here to the feature names before fetching them.
      # Also, the transformer features are stored in an array, so we take the first (and only) element.
      params[target] = {feat: getattr(transformer, self.camelToSnake(feat))[0] for feat in self._features}
      params[target]['model'] = transformer
    return params

  def getClusteringValues(self, nameTemplate: str, requests: list, params: dict) -> dict:
    """
      Provide the characteristic parameters of this ROM for clustering with other ROMs
      @ In, nameTemplate, str, formatting string template for clusterable params (target, metric id)
      @ In, requests, list, list of requested attributes from this ROM
      @ In, params, dict, parameters from training this ROM
      @ Out, features, dict, params as {paramName: value}
    """
    features = {}
    requestedFeatures = set(self._features).intersection(set(requests))
    for target, info in params.items():
      for feat in requestedFeatures:
        value = info[feat]
        key = nameTemplate.format(target=target, metric=self.name, id=feat)
        features[key] = value
    return features

  def getParamsAsVars(self, params):
    """
      Map characterization parameters into flattened variable format
      @ In, params, dict, trained parameters (as from fit)
      @ Out, rlz, dict, realization-style response
    """
    rlz = {}
    for target, info in params.items():
      base = f'{self.name}__{target}'
      for feat in self._features:
        rlz[f'{base}__{feat}'] = info[feat]
    return rlz

  def writeXML(self, writeTo, params):
    """
      Allows the engine to put whatever it wants into an XML to print to file.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, params, dict, trained parameters as from self.fit
      @ Out, None
    """
    for target, info in params.items():
      base = xmlUtils.newNode(target)
      # Add features as subnodes
      features = {k: v for k, v in info.items() if k in self._features}
      for featureName, featureValue in features.items():
        base.append(xmlUtils.newNode(featureName, text=featureValue))
      writeTo.append(base)

  @staticmethod
  def camelToSnake(camelName):
    """
      Converts a parameter name from camel case to snake case with a trailing underscore. Parameter
      names for scikit-learn transformers follow the convention of being snake case with a trailing
      underscore (e.g. "scale_" or "data_min_"). This method converts these names to camel case
      (e.g. "scale" or "dataMin") to align with the convention used in RAVEN.
      @ In, camelName, str, parameter name in camel case
      @ Out, paramName, str, parameter name
    """
    paramName = ''.join(['_' + c.lower() if c.isupper() else c for c in camelName]).lstrip('_') + '_'
    return paramName
