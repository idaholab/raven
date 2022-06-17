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
Created on Sept 5 2017

@author: wangc
"""
#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3-------------------------------------------

################################################################################
from ..EntityFactoryBase import EntityFactory
from ..utils import utils
from .CrossValidation import CrossValidation
from .SklCrossValidation import SciKitLearn
## [ Add new class here ]

factory = EntityFactory('CrossValidation')
factory.registerAllSubtypes(CrossValidation)
