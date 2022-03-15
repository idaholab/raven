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
  Created on May 6, 2021
  @author: alfoa
  extracted from alfoa (2/16/2013) Steps.py
"""
from ..EntityFactoryBase import EntityFactory

################################################################################
# Forward samplers
from .Step import Step
from .SingleRun import SingleRun
from .MultiRun import MultiRun
from .PostProcess import PostProcess
from .IOStep import IOStep
from .RomTrainer import RomTrainer

factory = EntityFactory('Step')
factory.registerAllSubtypes(Step)
