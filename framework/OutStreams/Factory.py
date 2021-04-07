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
  Created on April 5, 2016
  @author: maljdp
  extracted from alfoa (11/14/2013) OutStreamManager.py
"""

from EntityFactoryBase import EntityFactory

from utils import utils

from .OutStreamBase import OutStreamBase
from .FilePrint import FilePrint
from .GeneralPlot import GeneralPlot
# from .DataMining import DataMining
# from .VDCComparison import VDCComparison

factory = EntityFactory('OutStreamBase')
factory.registerType('Print', FilePrint)
factory.registerType('Plot', GeneralPlot)
