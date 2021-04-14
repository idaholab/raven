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
  Factory interface for returning classes and instances from the
  validation algorithms pool.

  Created on April 04, 2021

  @ author: alfoa
"""
from EntityFactoryBase import EntityFactory

from .ValidationBase import ValidationBase
from .Probabilistic import Probabilistic
#from .DSS import DSS
#from .PCM import PCM
#from .Representativity import Representativity

factory = EntityFactory('ValidationBase')
# TODO map lower case to upper case, because of silly ROM namespace problems
factory.registerAllSubtypes(ValidationBase)
