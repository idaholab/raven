import sys
import math
import distribution1D
import raventools
# initialize distribution container
distcont  = distribution1D.DistributionContainer.Instance()
# initialize decay heat curve
DecayHeatScalingFactor     = raventools.decayHeat(1,1,3600*24*30*8,0.74)
# intialize pump Coast Down curves
# PumpCoastDown acts on the Head of the Pumps
PumpCoastDown              = raventools.pumpCoastdown(22.5,8.9)
# PumpCoastDownSec acts
PumpCoastDownSec           = raventools.pumpCoastdown(10.5,1)


def initial_function(monitored, controlled, auxiliary):

    # Nominal/initial values
    # N.B. If restart run, this function is not called
    auxiliary.InitialHead                      = controlled.Head_PumpB
    auxiliary.initialInletSecPress             = controlled.high_pressure_secondary_A
    return

def control_function(monitored, controlled, auxiliary):
    # Random on following variables
    # if you want to randomize the Diesel Generator Back Up time, un-comment the following two lines
    #if monitored.time_step == 1:
        # Random on following variables
        
        #random_n_1 = distcont.random()
        #auxiliary.DeltaTimeScramToAux = distcont.randGen('auxBackUpTimeDist',random_n_1)       
    
    if monitored.time>=auxiliary.scram_start_time:
        auxiliary.ScramStatus = True
        print('SCRAM')
    else:
        auxiliary.ScramStatus = False
        print('OPERATIONAL STATE')
    #
    if auxiliary.ScramStatus: #we are in scram     
        #primary pump B
        if controlled.Head_PumpB>1.e-4*8.9:
            random_n_3 = distcont.random()
            noise_b = distcont.randGen('noise',random_n_3)
            #print('NOISE**********' + str(noise_b))
            if monitored.time<(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux): # not yet auxiliary system up
                controlled.Head_PumpB = PumpCoastDown.flowrateCalculation(monitored.time-auxiliary.scram_start_time) + noise_b 
                if controlled.Head_PumpB < 0:
                  controlled.Head_PumpB = 0
                controlled.friction1_SC_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
                controlled.friction2_SC_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
                controlled.friction1_CL_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
                controlled.friction2_CL_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
            else: #system up
                if auxiliary.init_exp_frict:
                    auxiliary.friction_time_start_exp = controlled.friction1_SC_B
                    auxiliary.init_exp_frict = False 
                if controlled.Head_PumpB <= 0.05*8.9:
                    controlled.Head_PumpB = 0.05*8.9  
                    if controlled.friction1_SC_B > 0.1:
                        controlled.friction1_SC_B = auxiliary.friction_time_start_exp*math.exp(-(monitored.time-(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux))/4.0)                         
                        controlled.friction2_SC_B = controlled.friction1_SC_B                        
                        controlled.friction1_CL_B = controlled.friction1_SC_B
                        controlled.friction2_CL_B = controlled.friction1_SC_B
                    else:
                        controlled.friction1_SC_B = 0.1                    
                        controlled.friction2_SC_B = 0.1                        
                        controlled.friction1_CL_B = 0.1
                        controlled.friction2_CL_B = 0.1 
                else:
                    controlled.Head_PumpB = PumpCoastDown.flowrateCalculation(monitored.time-auxiliary.scram_start_time) + noise_b 
                    if controlled.Head_PumpB < 0:
                        controlled.Head_PumpB = 0
                    if controlled.friction1_SC_B > 0.1:
                        controlled.friction1_SC_B = auxiliary.friction_time_start_exp*math.exp(-(monitored.time-(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux))/4.0)                         
                        controlled.friction2_SC_B = controlled.friction1_SC_B                        
                        controlled.friction1_CL_B = controlled.friction1_SC_B
                        controlled.friction2_CL_B = controlled.friction1_SC_B
                    else:
                        controlled.friction1_SC_B = 0.1                    
                        controlled.friction2_SC_B = 0.1                        
                        controlled.friction1_CL_B = 0.1
                        controlled.friction2_CL_B = 0.1
        else:
            random_n_3 = distcont.random()
            noise_b = distcont.randGen('noise',random_n_3)
            if monitored.time<(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux): # not yet auxiliary system up
                controlled.Head_PumpB = 0 
                controlled.friction1_SC_B = 15000
                controlled.friction2_SC_B = 15000
                controlled.friction1_CL_B = 15000
                controlled.friction2_CL_B = 15000
            else:
                if auxiliary.init_exp_frict:
                    auxiliary.friction_time_start_exp = controlled.friction1_SC_B
                    auxiliary.init_exp_frict = False 
                if controlled.Head_PumpB <= 0.05*8.9:
                    controlled.Head_PumpB = 0.05*8.9 
                    if controlled.friction1_SC_B > 0.1:
                        controlled.friction1_SC_B = auxiliary.friction_time_start_exp*math.exp(-(monitored.time-(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux))/4.0)                         
                        controlled.friction2_SC_B = controlled.friction1_SC_B                        
                        controlled.friction1_CL_B = controlled.friction1_SC_B
                        controlled.friction2_CL_B = controlled.friction1_SC_B
                    else:
                        controlled.friction1_SC_B = 0.1                    
                        controlled.friction2_SC_B = 0.1                        
                        controlled.friction1_CL_B = 0.1
                        controlled.friction2_CL_B = 0.1
                else:
                    controlled.Head_PumpB = PumpCoastDown.flowrateCalculation(monitored.time-auxiliary.scram_start_time) + noise_b 
                    controlled.friction1_SC_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
                    controlled.friction2_SC_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
                    controlled.friction1_CL_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
                    controlled.friction2_CL_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
        #primary pump A        
        if controlled.Head_PumpA>1.e-4*8.9:
            random_n_4 = distcont.random()
            noise_a = distcont.randGen('noise',random_n_4)
            if monitored.time<(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux): # not yet auxiliary system up
                controlled.Head_PumpA = PumpCoastDown.flowrateCalculation(monitored.time-auxiliary.scram_start_time) + noise_a  
                if controlled.Head_PumpA < 0:
                    controlled.Head_PumpA = 0
                controlled.friction1_SC_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
                controlled.friction2_SC_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
                controlled.friction1_CL_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
                controlled.friction2_CL_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q 
            else:
                if controlled.Head_PumpA <= 0.05*8.9:
                    controlled.Head_PumpA = 0.05*8.9 
                    if controlled.friction1_SC_A > 0.1:
                        controlled.friction1_SC_A = auxiliary.friction_time_start_exp*math.exp(-(monitored.time-(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux))/4.0)
                        controlled.friction2_SC_A = controlled.friction1_SC_A                  
                        controlled.friction1_CL_A = controlled.friction1_SC_A
                        controlled.friction2_CL_A = controlled.friction1_SC_A
                    else:
                        controlled.friction1_SC_A = 0.1
                        controlled.friction2_SC_A = 0.1                  
                        controlled.friction1_CL_A = 0.1
                        controlled.friction2_CL_A = 0.1
                else:
                    if controlled.friction1_SC_A > 0.1:
                        controlled.Head_PumpA = PumpCoastDown.flowrateCalculation(monitored.time-auxiliary.scram_start_time) + noise_a 
                    if controlled.Head_PumpA < 0:
                        controlled.Head_PumpA = 0                        
                        controlled.friction1_SC_A = auxiliary.friction_time_start_exp*math.exp(-(monitored.time-(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux))/4.0)
                        controlled.friction2_SC_A = controlled.friction1_SC_A                  
                        controlled.friction1_CL_A = controlled.friction1_SC_A
                        controlled.friction2_CL_A = controlled.friction1_SC_A
                    else:
                        controlled.friction1_SC_A = 0.1
                        controlled.friction2_SC_A = 0.1                  
                        controlled.friction1_CL_A = 0.1
                        controlled.friction2_CL_A = 0.1
        else:
            random_n_4 = distcont.random()
            noise_a = distcont.randGen('noise',random_n_4)
            if monitored.time<(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux): # not yet auxiliary system up
                controlled.Head_PumpA = 0         
                controlled.friction1_SC_A = 15000
                controlled.friction2_SC_A = 15000
                controlled.friction1_CL_A = 15000
                controlled.friction2_CL_A = 15000
            else:
                if controlled.Head_PumpA <= 0.05*8.9:
                    controlled.Head_PumpA = 0.05*8.9  
                    if controlled.friction1_SC_A > 0.1:
                        controlled.friction1_SC_A = auxiliary.friction_time_start_exp*math.exp(-(monitored.time-(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux))/4.0)
                        controlled.friction2_SC_A = controlled.friction1_SC_A                  
                        controlled.friction1_CL_A = controlled.friction1_SC_A
                        controlled.friction2_CL_A = controlled.friction1_SC_A
                    else:
                        controlled.friction1_SC_A = 0.1
                        controlled.friction2_SC_A = 0.1                  
                        controlled.friction1_CL_A = 0.1
                        controlled.friction2_CL_A = 0.1
                else:
                    controlled.Head_PumpA = PumpCoastDown.flowrateCalculation(monitored.time-auxiliary.scram_start_time) + noise_a  
                    if controlled.Head_PumpA < 0:
                        controlled.Head_PumpA = 0
                    controlled.friction1_SC_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
                    controlled.friction2_SC_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
                    controlled.friction1_CL_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
                    controlled.friction2_CL_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q 

        #core power following decay heat curve     
        controlled.power_CH1 = auxiliary.init_Power_Fraction_CH1*DecayHeatScalingFactor.powerCalculation(monitored.time-auxiliary.scram_start_time)
        controlled.power_CH2 = auxiliary.init_Power_Fraction_CH2*DecayHeatScalingFactor.powerCalculation(monitored.time-auxiliary.scram_start_time)
        controlled.power_CH3 = auxiliary.init_Power_Fraction_CH3*DecayHeatScalingFactor.powerCalculation(monitored.time-auxiliary.scram_start_time)
        #secondary system replaced by auxiliary secondary system
    if monitored.time<(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux) and auxiliary.ScramStatus: # not yet auxiliary system up
        print('not yet auxiliary system up')
        controlled.MassFlowRateIn_SC_B = 2.432*PumpCoastDownSec.flowrateCalculation(monitored.time-auxiliary.scram_start_time)
        controlled.MassFlowRateIn_SC_A = 2.432*PumpCoastDownSec.flowrateCalculation(monitored.time-auxiliary.scram_start_time)
            #if controlled.MassFlowRateIn_SC_B <= 0.08:
        #controlled.MassFlowRateIn_SC_B = 0.08
            #if controlled.MassFlowRateIn_SC_A == 0.08:
            #controlled.MassFlowRateIn_SC_A = 0.08            
    if monitored.time>=(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux) and auxiliary.ScramStatus: # auxiliary system up
        print('auxiliary system up')
        controlled.MassFlowRateIn_SC_B = 2.432*0.05
        controlled.MassFlowRateIn_SC_A = 2.432*0.05
    if (monitored.avg_temp_clad_CH1>auxiliary.CladTempTreshold) or (monitored.avg_temp_clad_CH2>auxiliary.CladTempTreshold) or (monitored.avg_temp_clad_CH3>auxiliary.CladTempTreshold):
        auxiliary.CladDamaged = True
        raise NameError ('exit condition reached - failure of the clad')
    return 

