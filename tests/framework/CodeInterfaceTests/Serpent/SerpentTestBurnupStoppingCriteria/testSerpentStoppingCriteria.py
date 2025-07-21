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
# Developed by NuCube Energy, Inc.

def testSerpentStoppingCriteria(raven):
  """
    Method to return a bool logic to stop the simulation
    if a certain condition is realized
    @ In, raven, object, the raven container (variables are accesable via raven.VarName)
    @ Out, stoppingCriteria, bool, True if the simulation needs to be stopped, False otherwise
  """
  if raven.impKeff_0[-1] < 1.0:
    return True
  return False

