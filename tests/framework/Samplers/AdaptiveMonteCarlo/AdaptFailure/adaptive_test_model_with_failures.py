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
#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
# This is a simple analytic test function
# y1 = (x1-1)**2 + x2**2
# y2 = (x1+1)**2 + x2**2

def run(raven,Input):
  """
    Simple test function.
    @ In, raven, object, Raven object container
    @ In, Input, dict, variable information from Raven
    @ Out, None.
  """
  if raven.x2 < 0.0:
    # this guarantee that the first sample fails
    raise RuntimeError("Failure on demand!")
  raven.y1 = (raven.x1-1)**2 + raven.x2**2
  raven.y2 = (raven.x1+1)**2 + raven.x2**2
