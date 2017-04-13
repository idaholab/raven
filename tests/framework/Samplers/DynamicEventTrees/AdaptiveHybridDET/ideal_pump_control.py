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
def initial_function(monitored, controlled, auxiliary):
  # here we store some critical parameters that we want in the output
  auxiliary.depresSystemDistThreshold    = distributions.depresSystemDist.getVariable('ProbabilityThreshold')
  auxiliary.depressurizationOnTime       = distributions.depresSystemDist.inverseCdf(auxiliary.depresSystemDistThreshold)
  auxiliary.PressureFailureDistThreshold = distributions.PressureFailureDist.getVariable('ProbabilityThreshold')
  auxiliary.PressureFailureValue         = distributions.PressureFailureDist.inverseCdf(auxiliary.PressureFailureDistThreshold)
  auxiliary.endSimulation = False

def restart_function(monitored, controlled, auxiliary):
  # here we store some critical parameters that we want in the output
  auxiliary.depresSystemDistThreshold    = distributions.depresSystemDist.getVariable('ProbabilityThreshold')
  auxiliary.depressurizationOnTime       = distributions.depresSystemDist.inverseCdf(auxiliary.depresSystemDistThreshold)
  auxiliary.PressureFailureDistThreshold = distributions.PressureFailureDist.getVariable('ProbabilityThreshold')
  auxiliary.PressureFailureValue         = distributions.PressureFailureDist.inverseCdf(auxiliary.PressureFailureDistThreshold)
  auxiliary.endSimulation = False

def keep_going_function(monitored, controlled, auxiliary):
  if auxiliary.endSimulation:
    return False
  return True

def control_function(monitored, controlled, auxiliary):
  if auxiliary.systemFailed:
    auxiliary.endSimulation = True
    print("SYSTEM FAILED!!!!!")
  if auxiliary.depressurizationOn:
    print("DEPRESSURIZATION SYSTEM ON!!!!!")
    controlled.inlet_TDV_p_bc = controlled.inlet_TDV_p_bc*0.99
    if controlled.inlet_TDV_p_bc < 1.0e5:
      controlled.inlet_TDV_p_bc = 1.0e5
  if not auxiliary.depressurizationOn:
    controlled.inlet_TDV_p_bc = 1.0e5 + 0.021*1.0e5*monitored.time


def dynamic_event_tree(monitored, controlled, auxiliary):
  if monitored.time_step <= 1:
    return
  if distributions.PressureFailureDist.checkCdf(controlled.inlet_TDV_p_bc) and (not auxiliary.systemFailed) and (not auxiliary.depressurizationOn):
    auxiliary.systemFailed = True
    return
  if distributions.depresSystemDist.checkCdf(monitored.time) and (not auxiliary.systemFailed) and (not auxiliary.depressurizationOn):
    auxiliary.depressurizationOn = True
    return
