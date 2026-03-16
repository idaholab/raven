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


if __name__ == "__main__":
  from pytest_runner import run_module
  run_module(__file__)
