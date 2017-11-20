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
The DataObjects module includes the different type of data representations
available in RAVEN

Created on September 16, 2015
@author: maljdp
supercedes DataObjects.py from alfoa (2/16/2013)
"""

from __future__ import absolute_import

# These lines ensure that we do not have to do something like:
# 'from DataObjects.Data import Data' outside of this submodule
#from .Data import Data, NotConsistentData, ConstructError
#from .PointSet import PointSet
#from .HistorySet import HistorySet
from .XDataSet import DataObject as Data
from .XDataSet import DataSet
from .XPointSet import PointSet
from .XHistorySet import HistorySet

from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import returnClass

# We should not really need this as we do not use wildcard imports
__all__ = ['Data','DataSet','PointSet','HistorySet']
