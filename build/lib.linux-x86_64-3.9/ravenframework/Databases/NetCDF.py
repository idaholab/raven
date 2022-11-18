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
Created on April 9, 2013

@author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

import os
import numpy as np
import xarray as xr

from ..utils import xmlUtils, mathUtils
from .Database import DataBase

class NetCDF(DataBase):
  """
    Stores data in netCDF format
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, spec, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    spec = super(NetCDF, cls).getInputSpecification()
    spec.description = r"""File storage format based on NetCDF4 protocol, which is natively compatible
                       with xarray DataSets used in RAVEN DataObjects."""

    return spec

  def __init__(self):
    """
      Constructor
      @ In, runInfoDict, dict, info from RunInfo block
      @ Out, None
    """
    super().__init__()
    self.printTag = 'DATABASE-NetCDF'  # For printing verbosity labels
    self._format = 'netcdf4'  # writing format for disk
    self._extension = '.nc'

  def saveDataToFile(self, source):
    """
      Saves the given data as database to file.
      @ In, source, DataObjects.DataObject, object to write to file
      @ Out, None
    """
    ds, meta = source.getData()
    # we actually just tell the DataSet to write out as netCDF
    path = self.get_fullpath()
    # TODO set up to use dask for on-disk operations
    # convert metadata into writeable
    for key, xml in meta.items():
      ds.attrs[key] = xmlUtils.prettify(xml.getRoot())
    # get rid of "object" types
    for var in ds:
      if ds[var].dtype == np.dtype(object):
        # is it a string?
        if mathUtils.isAString(ds[var].values[0]):
          ds[var] = ds[var].astype(str)
    # is there existing data? Read it in and merge it, if so
    # -> we've already wiped the file in initializeDatabase if it's in write mode
    if os.path.isfile(path):
      exists = xr.load_dataset(path)
      if 'RAVEN_sample_ID' in exists:
        floor = int(exists['RAVEN_sample_ID'].values[-1]) + 1
        new = ds['RAVEN_sample_ID'].values + floor
        ds = ds.assign_coords(RAVEN_sample_ID=new)
      # NOTE order matters! This preserves the sampling order in which data was inserted
      #      into this database
      ds = xr.concat((exists, ds), 'RAVEN_sample_ID')
    # if this is open somewhere else, we can't write to it
    # TODO is there a way to check if it's writable? I can't find one ...
    try:
      ds.to_netcdf(path, engine=self._format)
    except PermissionError:
      self.raiseAnError(PermissionError, f'NetCDF file "{path}" denied RAVEN permission to write! Is it open in another program?')

  def loadIntoData(self, target):
    """
      Loads this database into the target data object
      @ In, target, DataObjects.DataObjet, object to write data into
      @ Out, None
    """
    # the main data
    # NOTE: DO NOT use open_dataset unless you wrap it in a "with xr.open_dataset(f) as ds"!
    # -> open_dataset does NOT close the file object after loading!
    # -> however, load_dataset fully loads the ds into memory and closes the file.
    ds = xr.load_dataset(self.get_fullpath(), engine=self._format)
    # the meta data, convert from string to xml
    meta = dict((key, xmlUtils.staticFromString(val)) for key, val in ds.attrs.items())
    # set D.O. properties
    target.setData(ds, meta)

  def addRealization(self, rlz):
    """
      Adds a "row" (or "sample") to this database.
      This is the method to add data to this database.
      Note that rlz can include many more variables than this database actually wants.
      Before actually adding the realization, data is formatted for this data object.
      @ In, rlz, dict, {var:val} format where
                         "var" is the variable name as a string,
                         "val" is either a float or a np.ndarray of values.
      @ Out, None
    """
    # apparently we're storing samples!
    # -> do we already have data present?
    path = self.get_fullpath()
    if os.path.isfile(path):
      # load data as 100 sample chunks, lazily (not into memory)
      # -> using the argument "chunks" triggers the lazy loading using dask
      # existing = xr.open_dataset(path, chunks={'RAVEN_sample_ID': 100}) # TODO user option
      existing = True
      with xr.open_dataset(path) as ds: # autocloses at end of scope
        counter = int(ds.RAVEN_sample_ID.values[-1]) + 1
    else:
      existing = None
      counter = 0
    # create DS from realization # TODO make a feature of the Realization object
    indexMap = rlz.get('_indexMap', [{}])[0]
    indices = list(set().union(*(set(x) for x in indexMap.values())))
    # verbose but slower
    xarrs = {}
    for var in rlz:
      if var == '_indexMap' or var in indices + ['SampledVars', 'SampledVarsPb', 'crowDist', 'SamplerType']:
        continue
      if self.variables is not None and var not in self.variables:
        continue
      vals = rlz[var]
      dims = indexMap.get(var, [])
      if not dims and len(vals) == 1:
        vals = vals[0]
      coords = dict((idx, rlz[idx]) for idx in indexMap.get(var, []))
      xarrs[var] = xr.DataArray(vals, dims=dims, coords=coords).expand_dims(dim={'RAVEN_sample_ID': [counter]})
    rlzDS = xr.Dataset(xarrs)
    if existing:
      with xr.open_dataset(path) as ds: # autocloses at end of scope
        # after research, best approach is concatenating xr.DataSet along RAVEN_sample_ID dim
        new = xr.concat((ds, rlzDS), dim='RAVEN_sample_ID')
    else:
      new = rlzDS
    new.to_netcdf(path) # TODO would appending instead of writing work for new samples? I doubt it.
