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
  Unit tests for gaUtils helpers (dataset/array conversion and finite gene bounds for real-coded operators).
"""
import numpy as np
import xarray as xr

from ravenframework.utils import gaUtils


def test_dataset_to_data_array_shape(ga_dataset):
  data = gaUtils.datasetToDataArray(ga_dataset['dataset'], ga_dataset['variables'])
  assert isinstance(data, xr.DataArray)
  assert data.dims == ('chromosome', 'Gene')
  assert list(data.coords['Gene'].values) == ga_dataset['variables']
  np.testing.assert_allclose(data.values, np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_data_array_to_dict_roundtrip(ga_dataset):
  data = gaUtils.datasetToDataArray(ga_dataset['dataset'], ga_dataset['variables'])
  row = data.sel(chromosome=1)
  point = gaUtils.dataArrayToDict(row)
  assert point == {'x1': 2.0, 'x2': 4.0}


def _unboundedNormal():
  """
    A real, unbounded Normal(0, 1) distribution, built through its input specification.
    @ In, None
    @ Out, dist, Normal, the initialized distribution (bounds stored as +/- sys.float_info.max)
  """
  import os, sys
  import xml.etree.ElementTree as ET
  ravenDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)] + [os.pardir] * 4)))
  if ravenDir not in sys.path:
    sys.path.append(ravenDir)
  from ravenframework.utils.utils import find_crow
  find_crow(os.path.join(ravenDir, 'framework'))
  from ravenframework import MessageHandler, Distributions
  mh = MessageHandler.MessageHandler()
  mh.initialize({'verbosity': 'silent'})
  node = ET.Element('Normal', {'name': 'unbounded'})
  for tag, text in (('mean', '0'), ('sigma', '1')):
    ET.SubElement(node, tag).text = text
  dist = Distributions.factory.returnInstance('Normal')
  dist.setMessageHandler(mh)
  spec = dist.getInputSpecification()()
  spec.parseNode(node)
  dist._handleInput(spec)
  return dist


def test_unbounded_distribution_gives_finite_gene_bounds():
  """+/- sys.float_info.max bounds are treated as unbounded: the ppf range is used, not the sentinel."""
  dist = _unboundedNormal()
  low, high = gaUtils.finiteGeneBounds(dist, 0.1, -0.2)
  assert np.isfinite(high - low)
  assert -10.0 < low < -4.0 and 4.0 < high < 10.0


def test_real_coded_operators_stay_finite_on_unbounded_distribution():
  """SBX crossover and polynomial mutation on an unbounded Normal produce no NaN/inf genes."""
  from ravenframework.Optimizers.crossOverOperators import crossovers
  from ravenframework.Optimizers.mutators import mutators
  from ravenframework.utils import randomUtils
  randomUtils.randomSeed(1234)
  dist = _unboundedNormal()
  variables = ['x1', 'x2']
  distDict = {var: dist for var in variables}
  parents = xr.DataArray(np.array([[0.3, -1.2], [-0.8, 0.5], [1.1, 0.0], [-0.1, 2.0]]),
                         dims=['chromosome', 'Gene'], coords={'chromosome': np.arange(4), 'Gene': variables})
  children = crossovers.sbxCrossover(parents, variables=variables, crossoverProb=1.0, distDict=distDict)
  assert np.all(np.isfinite(children.values))
  mutated = mutators.polynomialMutator(children, distDict, mutationProb=1.0, variables=variables)
  assert np.all(np.isfinite(mutated.values))


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
