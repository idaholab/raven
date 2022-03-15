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
Created on Feb 7, 2013
@author: alfoa
This python module performs the loading of data from csv files
"""
import numpy as np
import pandas as pd

from .BaseClasses import MessageUser

class CsvLoader(MessageUser):
  """
    Class aimed to load the CSV files
  """
  acceptableUtils = ['pandas', 'numpy']

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.type = 'CsvLoader'               # naming type for this class
    self.printTag = self.type             # message handling representation
    self.allOutParam = False              # all output parameters?
    self.allFieldNames = []               # "header" of the CSV file

  def loadCsvFile(self, myFile, nullOK=None, utility='pandas'):
    """
      Function to load a csv file into realization format
      It also retrieves the headers
      The format of the csv must be comma-separated (pandas readable)
      @ In, myFile, string, Input file name (absolute path)
      @ In, nullOK, bool, indicates if null values are acceptable
      @ In, utility, str, indicates which utility should be used to load the csv
      @ Out, loadCsvFile, pandas.DataFrame or numpy.ndarray, the loaded data
    """
    if utility == 'pandas':
      return self._loadCsvPandas(myFile, nullOK=nullOK)
    elif utility == 'numpy':
      return self._loadCsvNumpy(myFile, nullOK=nullOK)
    else:
      self.raiseAnError(RuntimeError, f'Unrecognized CSV loading utility: "{utility}"')

  def _loadCsvPandas(self, myFile, nullOK=None):
    """
      Function to load a csv file into realization format
      It also retrieves the headers
      The format of the csv must be comma-separated (pandas readable)
      @ In, myFile, string, Input file name (absolute path)
      @ In, nullOK, bool, indicates if null values are acceptable
      @ Out, df, pandas.DataFrame, the loaded data
    """
    # first try reading the file
    try:
      df = pd.read_csv(myFile)
    except pd.errors.EmptyDataError:
      # no data in file
      self.raiseAWarning(f'Tried to read data from "{myFile}", but the file is empty!')
      return
    else:
      self.raiseADebug(f'Reading data from "{myFile}"')
    # check for NaN contents -> this isn't allowed in RAVEN currently, although we might need to change this for ND
    if (not nullOK) and (pd.isnull(df).values.sum() != 0):
      bad = pd.isnull(df).any(1).to_numpy().nonzero()[0][0]
      self.raiseAnError(IOError, f'Invalid data in input file: row "{bad+1}" in "{myFile}"')
    self.allFieldNames = list(df.columns)
    return df

  def _loadCsvNumpy(self, myFile, nullOK=None):
    """
      Function to load a csv file into realization format
      It also retrieves the headers
      The format of the csv must be comma-separated with all floats after header row
      @ In, myFile, string, Input file name (absolute path)
      @ In, nullOK, bool, indicates if null values are acceptable
      @ Out, data, np.ndarray, the loaded data
    """
    with open(myFile, 'rb') as f:
      head = f.readline().decode()
    self.allFieldNames = list(x.strip() for x in head.split(','))
    data = np.loadtxt(myFile, dtype=float, delimiter=',', ndmin=2, skiprows=1)
    return data

  def toRealization(self, data):
    """
      Converts data from the "loadCsvFile" format to a realization-style format (dictionary
      currently)
      @ In, data, pandas.DataFrame or np.ndarray, result of loadCsvFile
      @ Out, rlz, dict, realization
    """
    rlz = {}
    if isinstance(data, pd.DataFrame):
      rlz = dict((header, np.array(data[header])) for header in self.allFieldNames)
    elif isinstance(data, np.ndarray):
      rlz = dict((header, entry) for header, entry in zip(self.allFieldNames, data.T))
    return rlz

  def getAllFieldNames(self):
    """
      Function to get all field names found in the csv file
      @ In, None
      @ Out, allFieldNames, list, list of field names (headers)
    """
    return self.allFieldNames
