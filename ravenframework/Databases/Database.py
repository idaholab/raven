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
import copy
import os
import abc
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ..BaseClasses import BaseEntity, InputDataUser
from ..utils import InputData, InputTypes
# Internal Modules End------------------------------------------------------------------------------

class DataBase(BaseEntity, InputDataUser):
  """
    class to handle a database,
    Used to add and retrieve attributes and values from said database
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super().getInputSpecification()
    inputSpecification.addParam("directory", InputTypes.StringType)
    inputSpecification.addParam("filename", InputTypes.StringType)
    inputSpecification.addParam("readMode", InputTypes.makeEnumType("readMode", "readModeType", ["overwrite", "read"]), True)
    inputSpecification.addSub(InputData.parameterInputFactory("variables", contentType=InputTypes.StringListType))

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.database = None       # Database object
    self.exist = False         # does it exist?
    self.built = False         # is it built?
    self.filename = ""         # filename
    self.workingDir  = None    # RAVEN working dir
    self.databaseDir = None    # Database directory. Default = working directory.
    self.printTag = 'DATABASE' # For printing verbosity labels
    self.variables = None      # if not None, list of specific variables requested to be stored by user
    self._extension = '.db'    # filetype extension to use, if no filename given
    self.readMode = None

  def applyRunInfo(self, runInfo):
    """
      Use RunInfo
      @ In, runInfo, dict, run info
      @ Out, None
    """
    super().applyRunInfo(runInfo)
    self.workingDir = runInfo['WorkingDir']
    self.databaseDir = self.workingDir

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the database parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    if 'directory' in paramInput.parameterValues:
      self.databaseDir = copy.copy(paramInput.parameterValues['directory'])
      # if not absolute path, join with working directory
      if not os.path.isabs(self.databaseDir):
        self.databaseDir = os.path.abspath(os.path.join(self.workingDir,self.databaseDir))
    else:
      self.databaseDir = os.path.join(self.workingDir, 'DatabaseStorage')
    if 'filename' in paramInput.parameterValues:
      self.filename = copy.copy(paramInput.parameterValues['filename'])
    else:
      self.filename = self.name + self._extension
    # read the variables
    varNode = paramInput.findFirst("variables")
    if varNode is not None:
      self.variables =  varNode.value
    # read mode
    self.readMode = paramInput.parameterValues['readMode']
    self.raiseADebug(f'{self.type} "{self.name}" Read Mode is "{self.readMode}".')
    if self.readMode == 'overwrite':
      # check if self.databaseDir exists or create in case not
      if not os.path.isdir(self.databaseDir):
        os.makedirs(self.databaseDir, exist_ok=True)
    # get full path
    fullpath = self.get_fullpath()
    if os.path.isfile(fullpath):
      if self.readMode == 'read':
        self.exist = True
      elif self.readMode == 'overwrite':
        self.exist = False
      self.initializeDatabase()
    else:
      # file does not exist in path
      if self.readMode == 'read':
        self.raiseAnError(IOError, f'Requested to read from database, but it does not exist at "{fullpath}"; '+
                          'The path to the database must be either absolute, or relative to <workingDir>!')
      self.exist = False
      self.initializeDatabase()
    self.raiseAMessage(f'Database is located at "{fullpath}"')

  def initialize(self, *args, **kwargs):
    """
      Initialization for data object, if any.
      @ In, args, list, ordered arguments
      @ In, kwargs, dict, keyword arguments
      @ Out, None
    """

  def initializeDatabase(self):
    """
      Initialize underlying database object.
      @ In, None
      @ Out, None
    """
    # if in overwrite mode, remove existing database
    if self.readMode == 'overwrite':
      path = self.get_fullpath()
      if os.path.exists(path):
        os.remove(path)

  def get_fullpath(self):
    """
      Getter for full file path
      @ In, None
      @ Out, path, str, full path to db
    """
    path = os.path.join(self.databaseDir, self.filename)

    return path

  @abc.abstractmethod
  def saveDataToFile(self, source):
    """
      Saves the given data as database to file.
      @ In, source, DataObjects.DataObject, object to write to file
      @ Out, None
    """

  @abc.abstractmethod
  def loadIntoData(self, target):
    """
      Loads this database into the target data object
      @ In, target, DataObjects.DataObjet, object to write data into
      @ Out, None
    """

  @abc.abstractmethod
  def addRealization(self, rlz):
    """
      Adds a "row" (or "sample") to this database.
      This is the method to add data to this database.
      Note that rlz can include many more variables than this database actually wants.
      @ In, rlz, dict, {var:val} format where
                         "var" is the variable name as a string,
                         "val" is either a float or a np.ndarray of values.
      @ Out, None
    """

  @abc.abstractmethod
  def allRealizations(self):
    """
      Casts this database as an xr.Dataset.
      Efficiency note: this is the slowest part of typical data collection.
      @ In, None
      @ Out, allData, list of arrays, all the data from this data object.
    """
