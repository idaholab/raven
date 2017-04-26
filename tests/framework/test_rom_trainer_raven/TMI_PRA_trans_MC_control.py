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
import sys
import math
import distribution1D

def restart_function(monitored, controlled, auxiliary):
  auxiliary.scram_start_time = 101.0
  random_n_1 = stochasticEnv.random()
  auxiliary.DG1recoveryTime = 100.0 + distributions.crew1DG1.inverseCdf(random_n_1)

  random_n_2 = stochasticEnv.random()
  auxiliary.DG2recoveryTime = auxiliary.DG1recoveryTime * distributions.crew1DG2CoupledDG1.inverseCdf(random_n_2)

  random_n_3 = stochasticEnv.random()
  auxiliary.SecPGrecoveryTime = 400.0 + distributions.crewSecPG.inverseCdf(random_n_3)

  random_n_4 = stochasticEnv.random()
  auxiliary.CladTempTreshold = distributions.CladFailureDist.inverseCdf(random_n_4)

  random_n_5 = stochasticEnv.random() # primary offsite power recovery
  auxiliary.PrimPGrecoveryTime = distributions.PrimPGrecovery.inverseCdf(random_n_5)
  auxiliary.DeltaTimeScramToAux = min(auxiliary.DG1recoveryTime+auxiliary.DG2recoveryTime , auxiliary.SecPGrecoveryTime, auxiliary.PrimPGrecoveryTime)

  auxiliary.auxAbsolute = auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux
  return

def control_function(monitored, controlled, auxiliary):

  if monitored.time>=(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux) and auxiliary.ScramStatus:
    auxiliary.AuxSystemUp =  True
  if (monitored.avg_temp_clad_CH1>auxiliary.CladTempTreshold) or (monitored.avg_temp_clad_CH2>auxiliary.CladTempTreshold) or (monitored.avg_temp_clad_CH3>auxiliary.CladTempTreshold):
    auxiliary.CladDamaged = True


  #if auxiliary.CladDamaged:
  #    raise NameError ('exit condition reached - failure of the clad')
  auxiliary.a_power_CH1 = controlled.power_CH1
  auxiliary.a_power_CH2 = controlled.power_CH2
  auxiliary.a_power_CH3 = controlled.power_CH3
  auxiliary.a_friction2_CL_B = controlled.friction2_CL_B
  auxiliary.a_friction1_CL_B = controlled.friction1_CL_B
  auxiliary.a_friction2_SC_B = controlled.friction2_SC_B
  auxiliary.a_friction1_SC_B = controlled.friction1_SC_B
  auxiliary.a_friction2_CL_A = controlled.friction2_CL_A
  auxiliary.a_friction1_CL_A = controlled.friction1_CL_A
  auxiliary.a_friction2_SC_A = controlled.friction2_SC_A
  auxiliary.a_friction1_SC_A = controlled.friction1_SC_A
  auxiliary.a_Head_PumpB     = controlled.Head_PumpB
  auxiliary.a_Head_PumpA     = controlled.Head_PumpA
  auxiliary.a_MassFlowRateIn_SC_B = controlled.MassFlowRateIn_SC_B
  auxiliary.a_MassFlowRateIn_SC_A = controlled.MassFlowRateIn_SC_A

  if monitored.time>=auxiliary.scram_start_time:
    auxiliary.ScramStatus = True
    print('SCRAM')
  else:
    auxiliary.ScramStatus = False
    print('OPERATIONAL STATE')
  #
  if auxiliary.ScramStatus:
    #we are in scram
    #primary pump B
    if auxiliary.a_Head_PumpB>1.e-4*8.9:
      if not auxiliary.AuxSystemUp:
        # not yet auxiliary system up
        auxiliary.a_Head_PumpB = tools.PumpCoastDown.compute(monitored.time-auxiliary.scram_start_time)
        if auxiliary.a_Head_PumpB < (1.e-4*8.9):
          auxiliary.a_Head_PumpB = 1.e-4*8.9
        auxiliary.a_friction1_SC_B = auxiliary.frict_m*auxiliary.a_Head_PumpB + auxiliary.frict_q
        auxiliary.a_friction2_SC_B = auxiliary.frict_m*auxiliary.a_Head_PumpB + auxiliary.frict_q
        auxiliary.a_friction1_CL_B = auxiliary.frict_m*auxiliary.a_Head_PumpB + auxiliary.frict_q
        auxiliary.a_friction2_CL_B = auxiliary.frict_m*auxiliary.a_Head_PumpB + auxiliary.frict_q
      else:
        #system up
        if auxiliary.init_exp_frict:
          auxiliary.friction_time_start_exp = auxiliary.a_friction1_SC_B
          auxiliary.init_exp_frict = False
        if auxiliary.a_Head_PumpB <= 0.05*8.9:
          auxiliary.a_Head_PumpB = auxiliary.a_Head_PumpB*1.5
          if auxiliary.a_Head_PumpB > 0.05*8.9:
            auxiliary.a_Head_PumpB = 0.05*8.9
          if auxiliary.a_friction1_SC_B > 0.1:
            auxiliary.a_friction1_SC_B = auxiliary.friction_time_start_exp*math.exp(-(monitored.time-(auxiliary.scram_start_time++100.0))/4.0)
            auxiliary.a_friction2_SC_B = auxiliary.a_friction1_SC_B
            auxiliary.a_friction1_CL_B = auxiliary.a_friction1_SC_B
            auxiliary.a_friction2_CL_B = auxiliary.a_friction1_SC_B
          else:
            auxiliary.a_friction1_SC_B = 0.1
            auxiliary.a_friction2_SC_B = 0.1
            auxiliary.a_friction1_CL_B = 0.1
            auxiliary.a_friction2_CL_B = 0.1
        else:
          auxiliary.a_Head_PumpB = tools.PumpCoastDown.compute(monitored.time-auxiliary.scram_start_time)
          if auxiliary.a_Head_PumpB < (1.e-4*8.9):
            auxiliary.a_Head_PumpB = 1.e-4*8.9
          if auxiliary.a_friction1_SC_B > 0.1:
            auxiliary.a_friction1_SC_B = auxiliary.friction_time_start_exp*math.exp(-(monitored.time-(auxiliary.scram_start_time++100.0))/4.0)
            auxiliary.a_friction2_SC_B = auxiliary.a_friction1_SC_B
            auxiliary.a_friction1_CL_B = auxiliary.a_friction1_SC_B
            auxiliary.a_friction2_CL_B = auxiliary.a_friction1_SC_B
          else:
            auxiliary.a_friction1_SC_B = 0.1
            auxiliary.a_friction2_SC_B = 0.1
            auxiliary.a_friction1_CL_B = 0.1
            auxiliary.a_friction2_CL_B = 0.1
    else:
      if not auxiliary.AuxSystemUp:
        # not yet auxiliary system up
        auxiliary.a_Head_PumpB = 1.e-4*8.9
        auxiliary.a_friction1_SC_B = 15000
        auxiliary.a_friction2_SC_B = 15000
        auxiliary.a_friction1_CL_B = 15000
        auxiliary.a_friction2_CL_B = 15000
      else:
        if auxiliary.init_exp_frict:
          auxiliary.friction_time_start_exp = auxiliary.a_friction1_SC_B
          auxiliary.init_exp_frict = False
        if auxiliary.a_Head_PumpB <= 0.05*8.9:
          auxiliary.a_Head_PumpB = auxiliary.a_Head_PumpB*1.5
          if auxiliary.a_Head_PumpB > 0.05*8.9:
            auxiliary.a_Head_PumpB = 0.05*8.9
          if auxiliary.a_friction1_SC_B > 0.1:
            auxiliary.a_friction1_SC_B = auxiliary.friction_time_start_exp*math.exp(-(monitored.time-(auxiliary.scram_start_time++100.0))/4.0)
            auxiliary.a_friction2_SC_B = auxiliary.a_friction1_SC_B
            auxiliary.a_friction1_CL_B = auxiliary.a_friction1_SC_B
            auxiliary.a_friction2_CL_B = auxiliary.a_friction1_SC_B
          else:
            auxiliary.a_friction1_SC_B = 0.1
            auxiliary.a_friction2_SC_B = 0.1
            auxiliary.a_friction1_CL_B = 0.1
            auxiliary.a_friction2_CL_B = 0.1
        else:
          auxiliary.a_Head_PumpB = tools.PumpCoastDown.compute(monitored.time-auxiliary.scram_start_time)
          auxiliary.a_friction1_SC_B = auxiliary.frict_m*auxiliary.a_Head_PumpB + auxiliary.frict_q
          auxiliary.a_friction2_SC_B = auxiliary.frict_m*auxiliary.a_Head_PumpB + auxiliary.frict_q
          auxiliary.a_friction1_CL_B = auxiliary.frict_m*auxiliary.a_Head_PumpB + auxiliary.frict_q
          auxiliary.a_friction2_CL_B = auxiliary.frict_m*auxiliary.a_Head_PumpB + auxiliary.frict_q
    #primary pump A
    auxiliary.a_Head_PumpA     = auxiliary.a_Head_PumpB
    auxiliary.a_friction1_SC_A = auxiliary.a_friction1_SC_B
    auxiliary.a_friction2_SC_A = auxiliary.a_friction2_SC_B
    auxiliary.a_friction1_CL_A = auxiliary.a_friction1_CL_B
    auxiliary.a_friction2_CL_A = auxiliary.a_friction2_CL_B

    #core power following decay heat curve
    auxiliary.a_power_CH1 = auxiliary.init_Power_Fraction_CH1*tools.DecayHeatScalingFactor.compute(monitored.time-auxiliary.scram_start_time)
    auxiliary.a_power_CH2 = auxiliary.init_Power_Fraction_CH2*tools.DecayHeatScalingFactor.compute(monitored.time-auxiliary.scram_start_time)
    auxiliary.a_power_CH3 = auxiliary.init_Power_Fraction_CH3*tools.DecayHeatScalingFactor.compute(monitored.time-auxiliary.scram_start_time)
  #secondary system replaced by auxiliary secondary system
  if not auxiliary.AuxSystemUp and auxiliary.ScramStatus:
    # not yet auxiliary system up
    print('not yet auxiliary system up')
    auxiliary.a_MassFlowRateIn_SC_B = 2.542*tools.PumpCoastDownSec.compute(monitored.time-auxiliary.scram_start_time)
    auxiliary.a_MassFlowRateIn_SC_A = 2.542*tools.PumpCoastDownSec.compute(monitored.time-auxiliary.scram_start_time)
    if auxiliary.a_MassFlowRateIn_SC_A < (1.e-4*2.542):
      auxiliary.a_MassFlowRateIn_SC_A = 1.e-4*2.542
      auxiliary.a_MassFlowRateIn_SC_B = 1.e-4*2.542
  if auxiliary.AuxSystemUp and auxiliary.ScramStatus:
    # auxiliary system up
    print('auxiliary system up')
    auxiliary.a_MassFlowRateIn_SC_B = auxiliary.a_MassFlowRateIn_SC_B*1.5
    auxiliary.a_MassFlowRateIn_SC_A = auxiliary.a_MassFlowRateIn_SC_B
    if auxiliary.a_MassFlowRateIn_SC_B > 2.542*0.05:
      auxiliary.a_MassFlowRateIn_SC_B = 2.542*0.05
      auxiliary.a_MassFlowRateIn_SC_A = 2.542*0.05
  # we work on auxiliaries and we store them back into controlleds
  controlled.power_CH1 = auxiliary.a_power_CH1
  controlled.power_CH2 = auxiliary.a_power_CH2
  controlled.power_CH3 = auxiliary.a_power_CH3
  controlled.friction2_CL_B = auxiliary.a_friction2_CL_B
  controlled.friction1_CL_B = auxiliary.a_friction1_CL_B
  controlled.friction2_SC_B = auxiliary.a_friction2_SC_B
  controlled.friction1_SC_B = auxiliary.a_friction1_SC_B
  controlled.friction2_CL_A = auxiliary.a_friction2_CL_A
  controlled.friction1_CL_A = auxiliary.a_friction1_CL_A
  controlled.friction2_SC_A = auxiliary.a_friction2_SC_A
  controlled.friction1_SC_A = auxiliary.a_friction1_SC_A
  controlled.Head_PumpB = auxiliary.a_Head_PumpB
  controlled.Head_PumpA = auxiliary.a_Head_PumpA
  controlled.MassFlowRateIn_SC_B = auxiliary.a_MassFlowRateIn_SC_B
  controlled.MassFlowRateIn_SC_A = auxiliary.a_MassFlowRateIn_SC_A
  #if auxiliary.CladDamaged:
  #    raise NameError ('exit condition reached - failure of the clad')
  return
