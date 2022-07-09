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
  The Databases module includes efficient ways to serialize data to file.
"""

from __future__ import absolute_import

from ..utils import InputData

from .Database import DataBase as Database
from .HDF5 import HDF5
from .NetCDF import NetCDF

from .Factory import factory

class DatabasesCollection(InputData.ParameterInput):
  """
    Class for reading in a collection of databases
  """

DatabasesCollection.createClass("Databases")
DatabasesCollection.addSub(HDF5.getInputSpecification())
DatabasesCollection.addSub(NetCDF.getInputSpecification())

def returnInputParameter():
  """
    Returns the input specs for the desired classes
    @ In, None
    @ Out, returnInputParameter, InputData.ParameterInput, parsing class
  """
  return DatabasesCollection()
