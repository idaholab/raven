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
  Created on March 29, 2020

  @author: alfoa
"""
#Internal Modules---------------------------------------------------------------
from ..utils import importerUtils as im
from ..utils.utils import Object
#Internal Modules End-----------------------------------------------------------

#External Modules---------------------------------------------------------------
# for internal parallel
## TODO: REMOVE WHEN RAY AVAILABLE FOR WINDOWOS
_remote = None
if im.isLibAvail("ray"):
  from ray import remote as _remote
# end internal parallel module
#External Modules End-----------------------------------------------------------

class Parallel(object):
  """
    RAVEN parallel decorator
    It is used to mask ray parallel remote
    decorator and to allow for direct call of
    the underlying function (via _function attribute)
    i.e. :
    - if ray is available and needs to be used,
      the decorated function (or class) will need to be called as following:
      functionName.remote(*args, **kwargs)
    - if ray is available and direct call to the function is needed,
      the original function (or class) will need to be called as following:
      functionName.original_function(*args, **kwargs)
    - if ray is not available,
      the original function (or class) will need to be called as following:
      functionName(*args, **kwargs)
  """
  def __init__(self):
    """
      This is the constructor of the decorator for parallel execution
      @ In, None (it uses the _remote global variable)
      @ Out, None
    """
    self.decorator = _remote

  def __call__(self, func):
    """
      This method mimic the "decorator" method
      @ In, func, FunctionType or Class, the function or class to decorate
      @ Out, decorated, FunctionType, or Class, the decorated function or class
    """
    if self.decorator is None:
      # Return the function decorated with a wrapper
      # this is basically not decarate but we keep the same
      # approach for accessing to the original underlying function
      # in case of multi-threading
      decorated = Object()
    else:
      # decorate the function
      decorated = self.decorator(func)
    decorated.__dict__['original_function'] = func
    return decorated

