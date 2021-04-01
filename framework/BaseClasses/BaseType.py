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
Implements the common base class for all RAVEN objects (Entities and Interfaces)
Assures access to:
- InputData initialization
- MessageHandler
and similar utilities

Created on Mar 16, 2013
@author: crisr
"""

from .MessageUser import MessageUser
from .InputDataUser import InputDataUser

class BaseType(MessageUser, InputDataUser): # TODO add InputDataUser when all entities are converted
  """
    Implement the common base class for all RAVEN objects (Entities and Interfaces)
  """
  def __init__(self, **kwargs):
    """
      Construct.
      @ In, None
      @ Out, None
    """
    super().__init__(**kwargs)
    # identification
    self.name = None                     # name of this istance (alias)
    self.printTag = None                 # the tag that refers to this class in all the specific printing
    self.type = self.__class__.__name__  # specific type within this class

  def applyRunInfo(self, runInfo):
    """
      Allows access to the RunInfo data
      @ In, runInfo, dict, info from RunInfo
      @ Out, None
    """
    pass
