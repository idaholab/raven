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
import distribution1D
distcont  = distribution1D.DistributionContainer.instance()


def initial_function(monitored, controlled, auxiliary):
  print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
  auxiliary.dummy_for_branch = 0.0
  # Hybrid MonteCarlo (RAVEN way)
  controlled.pump_mass_flow_rate = controlled.pump_mass_flow_rate + controlled.pump_mass_flow_rate*distributions.testHybridMonteCarloDist2.getDistributionRandom()*0.1
  # Hybrid Grid (RAVEN way)
  controlled.inlet_TDV_p_bc = controlled.inlet_TDV_p_bc + controlled.inlet_TDV_p_bc*0.1*distributions.testHybridGridDist2.getDistributionRandom()
  # Hybrid LHS (RAVEN way)
  #controlled.pipe2_f = controlled.pipe2_f + controlled.pipe2_f*0.1*distributions.testHybridLHSDist2.getDistributionRandom()
  # Hybrid MonteCarlo (Sampled Variable directly from RAVEN FRAMEWORK)
  controlled.inlet_TDV_T_bc = controlled.inlet_TDV_T_bc + controlled.inlet_TDV_T_bc*0.1*auxiliary.testMCHybrid1
  # Hybrid Grid (Sampled Variable directly from RAVEN FRAMEWORK)
  controlled.outlet_TDV_p_bc = controlled.outlet_TDV_p_bc + controlled.outlet_TDV_p_bc*0.1*auxiliary.testGridHybrid1
  # Hybrid LHS (Sampled Variable directly from RAVEN FRAMEWORK)
  controlled.outlet_TDV_T_bc = controlled.outlet_TDV_T_bc + controlled.outlet_TDV_T_bc*0.1*auxiliary.testLHSHybrid1
  return

def control_function(monitored, controlled, auxiliary):
  print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
  if auxiliary.dummy_for_branch < 1.0:
    auxiliary.dummy_for_branch = auxiliary.dummy_for_branch + 0.25
  print('THRESHOLDDDDDD ' + str(distributions.zeroToOne.getVariable('ProbabilityThreshold')))
  return

def dynamic_event_tree(monitored, controlled, auxiliary):
  if distcont.checkCdf('zeroToOne',auxiliary.dummy_for_branch) and (not auxiliary.aBoolean) and monitored.time_step>1:
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    auxiliary.aBoolean = True
  return
