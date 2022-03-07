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
  Created on September 12, 2016
  @author: maljdp
  extracted from alfoa, cogljj, crisr (3/5/2013) JobHandler.py
"""
from ..EntityFactoryBase import EntityFactory

################################################################################
from ..utils import utils
from .Runner import Runner
from .DistributedMemoryRunner import DistributedMemoryRunner
from .InternalRunner import InternalRunner
from .PassthroughRunner import PassthroughRunner
from .SharedMemoryRunner import SharedMemoryRunner

class RunnerFactory(EntityFactory):
  """ Specific implementation for runners """
  def returnInstance(self, Type, funcArgs, func, **kwargs):
    """
      Returns an instance pointer from this module.
      @ In, Type, string, requested object
      @ In, caller, object, requesting object
      @ In, funcArgs, list, arguments to be passed as func(*funcArgs)
      @ In, func, method or function, function that needs to be run
      @ In, kwargs, dict, additional keyword arguments to constructor
      @ Out, returnInstance, instance, instance of the object
    """
    cls = self.returnClass(Type)
    instance = cls(funcArgs, func, **kwargs)
    return instance

factory = RunnerFactory('Runner')
factory.registerAllSubtypes(Runner)
