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
# for future compatibility with Python 3------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
# End compatibility block for Python 3--------------------------------------------------------------

# External Modules----------------------------------------------------------------------------------
import numpy as np
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ..h5py_interface_creator import hdf5Database as h5Data
from ..DataObjects import PointSet, HistorySet
from .Database import DataBase
# Internal Modules End------------------------------------------------------------------------------

class HDF5(DataBase):
  """
    class to handle h5py (hdf5) databases,
    Used to add and retrieve attributes and values from said database
  """

  def __init__(self):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    super().__init__()
    self.subtype = None
    self.type = 'HDF5'
    self._metavars = []
    self._allvars  = []
    self.printTag = 'DATABASE-HDF5'
    self._extension = '.h5'

  def __getstate__(self):
    """
      Overwrite state (for pickling)
      we do not pickle the HDF5 (C++) instance
      but only the info to reload it
      @ In, None
      @ Out, state, dict, the namespace state
    """
    # capture what is normally pickled
    state = self.__dict__.copy()
    # we pop the database instance and close it
    state.pop("database")
    self.database.closeDatabaseW()
    # what we return here will be stored in the pickle

    return state

  def __setstate__(self, newstate):
    """
      Set the state (for pickling)
      we do not pickle the HDF5 (C++) instance
      but only the info to reload it
      @ In, newstate, dict, the namespace state
      @ Out, None
    """
    self.__dict__.update(newstate)
    self.exist = True
    self.database = h5Data(self.name, self.databaseDir, self.filename, self.exist)

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the database parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)

  #####################
  # base API
  def initialize(self, gname, options=None):
    """
      Function to add an initial root group into the data base...
      This group will not contain a dataset but, eventually, only metadata
      @ In, gname, string, name of the root group
      @ In, options, dict, options (metadata muste be appended to the root group), Default =None
      @ Out, None
    """
    self.database.addGroupInit(gname,options)

  def initializeDatabase(self):
    """
      Initialize underlying database object.
      @ In, None
      @ Out, None
    """
    if self.database is not None:
      self.database.closeDatabaseW()
    super().initializeDatabase()
    self.database = h5Data(self.name, self.databaseDir, self.filename, self.exist, self.variables)

  def saveDataToFile(self, source):
    """
      Saves the given data as database to file.
      @ In, source, DataObjects.DataObject, object to write to file
      @ Out, None
    """
    if not isinstance(source, (PointSet, HistorySet)):
      self.raiseAnError(TypeError, 'RAVEN HDF5 Databases cannot currently handle N-Dimensional Datasets; ' +
                        f'use NetCDF instead. Received Dataset for database "{source.name}"')
    for r in range(len(source)):
      rlz = source.realization(r, unpackXArray=True)
      rlz = dict((var, np.atleast_1d(val)) for var, val in rlz.items())
      self.addRealization(rlz)

  def loadIntoData(self, target):
    """
      Loads this database into the target data object
      @ In, target, DataObjects.DataObjet, object to write data into
      @ Out, None
    """
    allRlz = self.allRealizations()
    for rlz in allRlz:
      target.addRealization(rlz)

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
    # realization must be a dictionary
    assert isinstance(rlz, dict)
    # prefix must be present
    if 'prefix' not in rlz:
      rlz['prefix'] = len(self.database)
    # check dimensionality
    if '_indexMap' in rlz:
      for var, dims in rlz['_indexMap'][0].items():
        if len(dims) > 1:
          self.raiseAnError(TypeError, 'RAVEN HDF5 Databases cannot currently handle N-Dimensional data; ' +
                            f'use NetCDF instead. Received ND data for variable "{var}": {dims}')
    self.database.addGroup(rlz)
    self.built = True

  #####################
  # utilities
  def getInitParams(self):
    """
      Function to get the initial values of the input parameters that belong to
      this class
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = DataBase.getInitParams(self)
    paramDict['exist'] = self.exist

    return paramDict

  def getEndingGroupNames(self):
    """
    Function to retrieve all the groups' names of the ending groups
    @ In, None
    @ Out, endingGroups, list, List of the ending groups' names
    """
    endingGroups = self.database.retrieveAllHistoryNames()

    return endingGroups

  def addExpectedMeta(self, keys, params={}):
    """
      Registers meta to look for in realizations.
      @ In, keys, set(str), keys to register
      @ In, params, dict, optional, {key:[indexes]}, keys of the dictionary are the variable names,
        values of the dictionary are lists of the corresponding indexes/coordinates of given variable
      @ Out, None
    """
    self.database.addExpectedMeta(keys,params)
    self.addMetaKeys(keys, params)

  def provideExpectedMetaKeys(self):
    """
      Provides the registered list of metadata keys for this entity.
      @ In, None
      @ Out, meta, tuple, (set(str),dict), expected keys (empty if none) and dictionary of expected keys with respect to their indexes, i.e. {keys:[indexes]}
    """
    return self.database.provideExpectedMetaKeys()

  def returnHistory(self,options):
    """
      Function to retrieve a history from the HDF5 database
      @ In, options, dict, options (metadata muste be appended to the root group)
      @ Out, tupleVar, tuple, tuple in which the first position is a numpy aray and the second is a dictionary of the metadata
      Note:
      # DET => a Branch from the tail (group name in attributes) to the head (dependent on the filter)
      # MC  => The History named ['group'] (one run)
    """
    # this retrieveHistory method seems to have been deprecated, does it exist somewhere/is it used anywhere?
    tupleVar = self.database.retrieveHistory(options['history'],options)

    return tupleVar

  def allRealizations(self):
    """
      Casts this database as an xr.Dataset.
      Efficiency note: this is the slowest part of typical data collection.
      @ In, None
      @ Out, allData, list of arrays, all the data from this data object.
    """
    allRealizationNames = self.database.retrieveAllHistoryNames()
    # instead to use a OrderedDict in the database, I sort the names here (it is much faster)
    allRealizationNames.sort()
    allData = [self.realization(name) for name in allRealizationNames]

    return allData

  def realization(self, index=None, matchDict=None, tol=1e-15):
    """
      Method to obtain a realization from the data, either by index (e.g. realization number) or matching value.
      Either "index" or "matchDict" must be supplied. (NOTE: now just "index" can be supplied)
      @ In, index, int or str, optional, number of row to retrieve (by index, not be "sample") or group name
      @ In, matchDict, dict, optional, {key:val} to search for matches
      @ In, tol, float, optional, tolerance to which match should be made
      @ Out, index, int, optional, index where found (or len(self) if not found), only returned if matchDict
      @ Out, rlz, dict, realization requested (None if not found)
    """
    # matchDict not implemented for Databases
    assert matchDict is None
    if (not self.exist) and (not self.built):
      self.raiseAnError(Exception, f'Can not retrieve a realization from Database {self.name} .It has not been built yet!')
    if type(index).__name__ == 'int':
      allRealizations = self.database.retrieveAllHistoryNames()
    if type(index).__name__ == 'int' and index > len(allRealizations):
      rlz = None
    else:
      rlz, _ = self.database._getRealizationByName(allRealizations[index] if type(index).__name__ == 'int' else index , {'reconstruct': False})

    return rlz
