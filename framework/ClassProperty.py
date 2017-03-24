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
    This allows for the construction of static constant class properties that
    any object can use, but they should not be allowed to modify
"""

## Custom class allows for the @ClassProperty decorator to be used to specify
## a class level, immutable, and static property

class ClassProperty(object):
  """
      A custom class providing the option to create a class level decorator
      (e.g. @ClassProperty) above a method to make it a static and immutable
      class variable
  """
  def __init__(self, func):
    """
        Constructor that will associate this ClassProperty to a particular
        method
        @ In, func, a function
        @ Out, None
    """
    self.func = func

  def __get__(self, inst, cls):
    """
        Overloaded getter that will ignores the instance and returns the class
        call to the ClassProperty associated function
        @ In, inst, instance object attempting to retrieve this ClassProperty
          (ignored, thus it can be undefined)
        @ In, cls, class attempting to retrieve this ClassProperty
        @ Out, variable, this will return whatever the ClassProperty's func
          returns
    """
    return self.func(cls)
