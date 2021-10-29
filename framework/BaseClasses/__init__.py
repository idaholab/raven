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
  Module defines the Base Classes used for standard objects in RAVEN.

  Created on March 19, 2021
  @author: talbpaul
  supercedes BaseClasses.py from alfoa (2013-03-16)
"""

# Design notes, see github #1486

# Make this namespace available at the framework level

from .InputDataUser import InputDataUser
from .MessageUser import MessageUser
from .BaseType import BaseType
from .BaseEntity import BaseEntity
from .PluginReadyEntity import PluginReadyEntity
from .Assembler import Assembler
from .BaseInterface import BaseInterface

