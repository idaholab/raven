import sys
import math
import distribution1D
import raventools


distcont  = distribution1D.DistributionContainer.Instance()
toolcont  = raventools.RavenToolsContainer.Instance()

#DecayHeatScalingFactor     = raventools.decayHeat(1,1,3600*24*30*8,0.064)
#PumpCoastDown              = raventools.pumpCoastdown(22.5,9.9)
#PumpCoastDownSec           = raventools.pumpCoastdown(22.5,1)

def initial_function(monitored, controlled, auxiliary):

#Nominal/initial values
    #auxiliary.InitialHead                      = controlled.Head_PumpB
    #auxiliary.initialInletSecPress             = controlled.high_pressure_secondary_A
    return

def control_function(monitored, controlled, auxiliary):
    if monitored.time_step == 1:
        # Random on following variables
        auxiliary.scram_start_time = 120.01
        random_n_1 = distcont.random()
        auxiliary.DG1recoveryTime   = 10.0 + distcont.randGen('crew1DG1',random_n_1) 
        
        random_n_2 = distcont.random()
        auxiliary.DG2recoveryTime   = 10.0 + auxiliary.DG1recoveryTime * distcont.randGen('crew1DG2CoupledDG1',random_n_2)
     
        random_n_3 = distcont.random()
        auxiliary.SecPGrecoveryTime = 20.0 + distcont.randGen('crewSecPG',random_n_3) 
    

    if monitored.time >= auxiliary.scram_start_time:
        auxiliary.ScramStatus = True
    else:
        auxiliary.ScramStatus = False
    
    
    auxiliary.DeltaTimeScramToAux = min(auxiliary.DG1recoveryTime+auxiliary.DG2recoveryTime , auxiliary.SecPGrecoveryTime)

    if monitored.time < (auxiliary.scram_start_time + auxiliary.DeltaTimeScramToAux):
        auxiliary.PowerStatus = 0
    else:
        auxiliary.PowerStatus = 1
     
    
    if auxiliary.ScramStatus: #we are in scram situation    
        #primary pump B
        if controlled.Head_PumpB>1.e-4*9.9:
            if monitored.time<(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux): # not yet auxiliary system up
                controlled.Head_PumpB = toolcont.compute('PumpCoastDown',monitored.time-auxiliary.scram_start_time) 
                controlled.friction1_SC_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
                controlled.friction2_SC_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
                controlled.friction1_CL_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
                controlled.friction2_CL_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
            else: #system up
                if auxiliary.init_exp_frict:
                    auxiliary.friction_time_start_exp = controlled.friction1_SC_B
                    auxiliary.init_exp_frict = False 
                if controlled.Head_PumpB <= 0.05*9.9:
                    controlled.Head_PumpB = 0.05*9.9
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
                    controlled.Head_PumpB = toolcont.compute('PumpCoastDown',monitored.time-auxiliary.scram_start_time)
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
            if monitored.time<(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux): # not yet auxiliary system up
                controlled.Head_PumpB = 0
                controlled.friction1_SC_B = 5000
                controlled.friction2_SC_B = 5000
                controlled.friction1_CL_B = 5000
                controlled.friction2_CL_B = 5000
            else:
                if auxiliary.init_exp_frict:
                    auxiliary.friction_time_start_exp = controlled.friction1_SC_B
                    auxiliary.init_exp_frict = False 
                if controlled.Head_PumpB <= 0.05*9.9:
                    controlled.Head_PumpB = 0.05*9.9
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
                    controlled.Head_PumpB = toolcont.compute('PumpCoastDown',monitored.time-auxiliary.scram_start_time)
                    controlled.friction1_SC_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
                    controlled.friction2_SC_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
                    controlled.friction1_CL_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
                    controlled.friction2_CL_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
        #primary pump A        
        if controlled.Head_PumpA>1.e-4*9.9:
            if monitored.time<(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux): # not yet auxiliary system up
                controlled.Head_PumpA = toolcont.compute('PumpCoastDown',monitored.time-auxiliary.scram_start_time) 
                controlled.friction1_SC_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
                controlled.friction2_SC_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
                controlled.friction1_CL_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
                controlled.friction2_CL_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q 
            else:
                if controlled.Head_PumpA <= 0.05*9.9:
                    controlled.Head_PumpA = 0.05*9.9
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
                        controlled.Head_PumpA = toolcont.compute('PumpCoastDown',monitored.time-auxiliary.scram_start_time) 
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
            if monitored.time<(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux): # not yet auxiliary system up
                controlled.Head_PumpA = 0       
                controlled.friction1_SC_A = 5000
                controlled.friction2_SC_A = 5000
                controlled.friction1_CL_A = 5000
                controlled.friction2_CL_A = 5000
            else:
                if controlled.Head_PumpA <= 0.05*9.9:
                    controlled.Head_PumpA = 0.05*9.9
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
                    controlled.Head_PumpA = toolcont.compute('PumpCoastDown',monitored.time-auxiliary.scram_start_time) 
                    controlled.friction1_SC_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
                    controlled.friction2_SC_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
                    controlled.friction1_CL_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
                    controlled.friction2_CL_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q 

        #core power following decay heat curve     
        controlled.power_CH1 = auxiliary.init_Power_Fraction_CH1*toolcont.compute('DecayHeatScalingFactor',monitored.time-auxiliary.scram_start_time)
        controlled.power_CH2 = auxiliary.init_Power_Fraction_CH2*toolcont.compute('DecayHeatScalingFactor',monitored.time-auxiliary.scram_start_time)
        controlled.power_CH3 = auxiliary.init_Power_Fraction_CH3*toolcont.compute('DecayHeatScalingFactor',monitored.time-auxiliary.scram_start_time)
        #secondary system replaced by auxiliary secondary system
    if monitored.time<(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux) and auxiliary.ScramStatus: # not yet auxiliary system up
        print('not yet auxiliary system up')
        if controlled.high_pressure_secondary_A >= auxiliary.InitialOutletSecPress: 
            if (auxiliary.InitialOutletSecPress + (auxiliary.initialInletSecPress - auxiliary.InitialOutletSecPress)*(1-toolcont.compute('PumpCoastDownSec',monitored.time-auxiliary.scram_start_time))) >= (auxiliary.InitialOutletSecPress+29.35152e+4):  
                 controlled.high_pressure_secondary_A = auxiliary.InitialOutletSecPress + (auxiliary.initialInletSecPress - auxiliary.InitialOutletSecPress)*(1-toolcont.compute('PumpCoastDownSec',monitored.time-auxiliary.scram_start_time))
                 print('controlled.high_pressure_secondary_A')
                 print(str(controlled.high_pressure_secondary_A))                            
            else:
                 controlled.high_pressure_secondary_A = auxiliary.InitialOutletSecPress+29.35152e+3
                 print('controlled.high_pressure_secondary_A')
                 print(str(controlled.high_pressure_secondary_A))
        else:
              controlled.high_pressure_secondary_A = auxiliary.InitialOutletSecPress+29.35152e+3 #otherwise just compensate hydro pressure
              print('controlled.high_pressure_secondary_A')
              print(str(controlled.high_pressure_secondary_A))
        if controlled.high_pressure_secondary_B >= auxiliary.InitialOutletSecPress:
            if (auxiliary.InitialOutletSecPress + (auxiliary.initialInletSecPress - auxiliary.InitialOutletSecPress)*(1-toolcont.compute('PumpCoastDownSec',monitored.time-auxiliary.scram_start_time))) >= (auxiliary.InitialOutletSecPress+29.35152e+4):   
                controlled.high_pressure_secondary_B = auxiliary.InitialOutletSecPress + (auxiliary.initialInletSecPress - auxiliary.InitialOutletSecPress)*(1-toolcont.compute('PumpCoastDownSec',monitored.time-auxiliary.scram_start_time))    
                print('controlled.high_pressure_secondary_B')               
                print(str(controlled.high_pressure_secondary_B))
            else:
                 controlled.high_pressure_secondary_B = auxiliary.InitialOutletSecPress+29.35152e+3    
                 print('controlled.high_pressure_secondary_B')
                 print(str(controlled.high_pressure_secondary_B))
        else:
            controlled.high_pressure_secondary_B = auxiliary.InitialOutletSecPress+29.35152e+3 #otherwise just compensate hydro pressure 
            print('controlled.high_pressure_secondary_B')   
            print(str(controlled.high_pressure_secondary_B))           
    if monitored.time>=(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux) and auxiliary.ScramStatus: # auxiliary system up
        print('auxiliary system up')
        controlled.high_pressure_secondary_A = (auxiliary.InitialOutletSecPress+29.35152e+3) + (auxiliary.initialInletSecPress - (auxiliary.InitialOutletSecPress+29.35152e+3) )*0.05       
        print('controlled.high_pressure_secondary_A') 
        print(str(controlled.high_pressure_secondary_A))
        controlled.high_pressure_secondary_B = (auxiliary.InitialOutletSecPress+29.35152e+3) + (auxiliary.initialInletSecPress -(auxiliary.InitialOutletSecPress+29.35152e+3))*0.05        
        print('controlled.high_pressure_secondary_B') 
        print(str(controlled.high_pressure_secondary_B))
    if (monitored.max_temp_clad_CH1>auxiliary.CladTempTreshold) or (monitored.max_temp_clad_CH2>auxiliary.CladTempTreshold) or (monitored.max_temp_clad_CH3>auxiliary.CladTempTreshold):
        auxiliary.CladDamaged = True
        raise NameError ('exit condition reached - failure of the clad')
    return 

