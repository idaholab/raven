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

def constrain(raven):
  """
    Constrain.
    @ In, raven, object, raven self
    @ Out, explicitConstraint, point ok or not?
  """
  v0 = raven.v0
  if v0 >= 130:
    return False
  return True

def implicitConstraint(raven):
  """
    Constrain.
    @ In, raven, object, raven self
    @ Out, implicitConstraint, point ok or not?
  """
  ymax = raven.ymax
  if ymax >= 250:
    return False
  return True
