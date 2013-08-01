import sys
import math
import distribution1D
import raventools
# initialize distribution container
distcont  = distribution1D.DistributionContainer.Instance()
toolcont  = raventools.RavenToolsContainer.Instance()

def initial_function(monitored, controlled, auxiliary):    
    return

def control_function(monitored, controlled, auxiliary):
    #print(monitored,controlled,auxiliary)
    #we use aux var in order to keep track of variables' values must be transfered from a calculation to the others
    if monitored.time_step == 1:
        random_n_1 = distcont.random()
        random_n_2 = distcont.random()
        #Rand is different on different machines, therefore, this is hardcoded.
        random_n_1 = 0.8401877171547095
        random_n_2 = 0.39438292681909304
        print("random_n_1 "+str(random_n_1)+" random_n_2 "+str(random_n_2))
        auxiliary.DeltaTimeScramToAux = distcont.randGen('auxBackUpTimeDist',random_n_1)         
        auxiliary.CladTempTresholdRNG = distcont.randGen('CladFailureDist',random_n_2)   
        print("DeltaTimeScramToAux "+str(auxiliary.DeltaTimeScramToAux)+
              " CladTempTresholdRNG "+str(auxiliary.CladTempTresholdRNG))
    
    if monitored.time>=(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux) and auxiliary.ScramStatus: 
        auxiliary.AuxSystemUp =  True
    if (monitored.avg_temp_clad_CH1>auxiliary.CladTempTreshold) or (monitored.avg_temp_clad_CH2>auxiliary.CladTempTreshold) or (monitored.avg_temp_clad_CH3>auxiliary.CladTempTreshold):
        auxiliary.CladDamaged = True
    if (monitored.avg_temp_clad_CH1>auxiliary.CladTempTresholdRNG) or (monitored.avg_temp_clad_CH2>auxiliary.CladTempTresholdRNG) or (monitored.avg_temp_clad_CH3>auxiliary.CladTempTresholdRNG):
        auxiliary.CladDamagedRNG = True

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
    if auxiliary.ScramStatus: #we are in scram     
        #primary pump B
        if auxiliary.a_Head_PumpB>1.e-4*8.9:
            if not auxiliary.AuxSystemUp: # not yet auxiliary system up
                auxiliary.a_Head_PumpB = toolcont.compute('PumpCoastDown',monitored.time-auxiliary.scram_start_time)
                if auxiliary.a_Head_PumpB < (1.e-4*8.9):
                    auxiliary.a_Head_PumpB = 1.e-4*8.9
                auxiliary.a_friction1_SC_B = auxiliary.frict_m*auxiliary.a_Head_PumpB + auxiliary.frict_q
                auxiliary.a_friction2_SC_B = auxiliary.frict_m*auxiliary.a_Head_PumpB + auxiliary.frict_q
                auxiliary.a_friction1_CL_B = auxiliary.frict_m*auxiliary.a_Head_PumpB + auxiliary.frict_q
                auxiliary.a_friction2_CL_B = auxiliary.frict_m*auxiliary.a_Head_PumpB + auxiliary.frict_q
            else: #system up
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
                    auxiliary.a_Head_PumpB = toolcont.compute('PumpCoastDown',monitored.time-auxiliary.scram_start_time) 
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
            if not auxiliary.AuxSystemUp: # not yet auxiliary system up
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
                    auxiliary.a_Head_PumpB = toolcont.compute('PumpCoastDown',monitored.time-auxiliary.scram_start_time) 
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
        auxiliary.a_power_CH1 = auxiliary.init_Power_Fraction_CH1*toolcont.compute('DecayHeatScalingFactor',monitored.time-auxiliary.scram_start_time)
        auxiliary.a_power_CH2 = auxiliary.init_Power_Fraction_CH2*toolcont.compute('DecayHeatScalingFactor',monitored.time-auxiliary.scram_start_time)
        auxiliary.a_power_CH3 = auxiliary.init_Power_Fraction_CH3*toolcont.compute('DecayHeatScalingFactor',monitored.time-auxiliary.scram_start_time)
    #secondary system replaced by auxiliary secondary system
    if not auxiliary.AuxSystemUp and auxiliary.ScramStatus: # not yet auxiliary system up
        print('not yet auxiliary system up')
        auxiliary.a_MassFlowRateIn_SC_B = 2.542*toolcont.compute('PumpCoastDownSec',monitored.time-auxiliary.scram_start_time) 
        auxiliary.a_MassFlowRateIn_SC_A = 2.542*toolcont.compute('PumpCoastDownSec',monitored.time-auxiliary.scram_start_time) 
        if auxiliary.a_MassFlowRateIn_SC_A < (1.e-4*2.542):
            auxiliary.a_MassFlowRateIn_SC_A = 1.e-4*2.542
            auxiliary.a_MassFlowRateIn_SC_B = 1.e-4*2.542
    if auxiliary.AuxSystemUp and auxiliary.ScramStatus: # auxiliary system up
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

