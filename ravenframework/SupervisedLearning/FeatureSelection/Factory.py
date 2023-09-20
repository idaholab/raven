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
  Created on Feb 18, 2022
  @author: Andrea Alfonsi
"""
from ...EntityFactoryBase import EntityFactory

################################################################################
## internal developed feature selection algorithms
from .RFE import RFE
from .VarianceThreshold import VarianceThreshold
################################################################################

factory = EntityFactory('FeatureSelection')
factory.registerType("RFE", RFE)
factory.registerType("VarianceThreshold", VarianceThreshold)
