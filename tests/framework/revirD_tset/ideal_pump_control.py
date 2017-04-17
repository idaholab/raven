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

def initial_function(monitored, controlled, auxiliary):
  print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
  mult = 1.01
  controlled.pipe1_Area = mult*controlled.pipe1_Area
  controlled.pipe1_Hw = mult*controlled.pipe1_Hw
  controlled.pipe2_Area = mult*controlled.pipe2_Area
  controlled.pipe2_Hw = mult*controlled.pipe2_Hw
  controlled.pump_mass_flow_rate = mult*controlled.pump_mass_flow_rate
  controlled.inlet_TDV_p_bc = mult*controlled.inlet_TDV_p_bc
  controlled.inlet_TDV_T_bc = mult*controlled.inlet_TDV_T_bc
  controlled.inlet_TDV_void_fraction_bc = mult*controlled.inlet_TDV_void_fraction_bc
  controlled.outlet_TDV_p_bc = mult*controlled.outlet_TDV_p_bc
  controlled.outlet_TDV_T_bc = mult*controlled.outlet_TDV_T_bc
  controlled.outlet_TDV_void_fraction_bc = mult*controlled.outlet_TDV_void_fraction_bc
  auxiliary.dummy_for_branch = 0.0

  return

def control_function(monitored, controlled, auxiliary):
  print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
  mult = 1.01
  if auxiliary.dummy_for_branch < 1.0:
    auxiliary.dummy_for_branch = auxiliary.dummy_for_branch + 0.25
  print('THRESHOLDDDDDD ' + str(distributions.zeroToOne.getVariable('ProbabilityThreshold')))
  controlled.pipe1_Area = mult*controlled.pipe1_Area
  controlled.pipe1_Hw = mult*controlled.pipe1_Hw
  controlled.pipe2_Area = mult*controlled.pipe2_Area
  controlled.pipe2_Hw = mult*controlled.pipe2_Hw
  controlled.pump_mass_flow_rate = mult*controlled.pump_mass_flow_rate
  controlled.inlet_TDV_p_bc = mult*controlled.inlet_TDV_p_bc
  controlled.inlet_TDV_T_bc = mult*controlled.inlet_TDV_T_bc
  controlled.inlet_TDV_void_fraction_bc = mult*controlled.inlet_TDV_void_fraction_bc
  controlled.outlet_TDV_p_bc = mult*controlled.outlet_TDV_p_bc
  controlled.outlet_TDV_T_bc = mult*controlled.outlet_TDV_T_bc
  controlled.outlet_TDV_void_fraction_bc = mult*controlled.outlet_TDV_void_fraction_bc
  return
