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
# 'from DataObjects.DataSet import DataSet' outside of this submodule
from .DataSet import DataObject as Data
from .DataSet import DataSet
from .PointSet import PointSet
from .HistorySet import HistorySet

from .Factory import factory
