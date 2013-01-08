import sys
import math
import distribution1D
import raventools
distcont  = distribution1D.DistributionContainer.Instance()
DecayHeatScalingFactor     = raventools.decayHeat(1,1,3600*24*30*8,0.064)
PumpCoastDown              = raventools.pumpCoastdown(22.5,9.9)
PumpCoastDownSec           = raventools.pumpCoastdown(22.5,1)

def initial_function(monitored, controlled, auxiliary):

#Nominal/initial values
    auxiliary.InitialHead                      = controlled.Head_PumpB
    auxiliary.initialInletSecPress             = controlled.high_pressure_secondary_A
    return

def control_function(monitored, controlled, auxiliary):

    #auxiliary.initialInletSecPress             = controlled.high_pressure_secondary_A
#    if monitored.time_step == 1:
#        controlled.rho_clad_CH1 =  controlled.rho_clad_CH1*100
#        controlled.rho_clad_CH2 =  controlled.rho_clad_CH1
#        controlled.rho_clad_CH3 =  controlled.rho_clad_CH1
#        controlled.rho_fuel_CH1 =  controlled.rho_fuel_CH1*100
#        controlled.rho_fuel_CH2 =  controlled.rho_fuel_CH1
#        controlled.rho_fuel_CH3 =  controlled.rho_fuel_CH1
#        controlled.rho_gap_CH1 =  controlled.rho_gap_CH1*100
#        controlled.rho_gap_CH2 =  controlled.rho_gap_CH1
#        controlled.rho_gap_CH3 =  controlled.rho_gap_CH1      
    if monitored.time>=auxiliary.scram_start_time:
        auxiliary.ScramStatus = True
    else:
        auxiliary.ScramStatus = False
    #
    if auxiliary.ScramStatus: #we are in scram situation    
        #primary pump B
        if controlled.Head_PumpB>1.e-4*9.9:
            controlled.Head_PumpB = PumpCoastDown.flowrateCalculation(monitored.time-auxiliary.scram_start_time) 
            controlled.friction1_SC_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
            controlled.friction2_SC_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
            controlled.friction1_CL_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
            controlled.friction2_CL_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
        else:
            controlled.Head_PumpB = 0
            controlled.friction1_SC_B = 10000
            controlled.friction2_SC_B = 10000 
            controlled.friction1_CL_B = 10000
            controlled.friction2_CL_B = 10000
        #primary pump A        
        if controlled.Head_PumpA>1.e-4*9.9:
            controlled.Head_PumpA = PumpCoastDown.flowrateCalculation(monitored.time-auxiliary.scram_start_time) 
            controlled.friction1_SC_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
            controlled.friction2_SC_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
            controlled.friction1_CL_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q
            controlled.friction2_CL_A = auxiliary.frict_m*controlled.Head_PumpA + auxiliary.frict_q   
        else:
            controlled.Head_PumpA = 0       
            controlled.friction1_SC_A = 10000
            controlled.friction2_SC_A = 10000
            controlled.friction1_CL_A = 10000
            controlled.friction2_CL_A = 10000 
        #core power following decay heat curve     
        controlled.power_CH1 = auxiliary.init_Power_Fraction_CH1*DecayHeatScalingFactor.powerCalculation(monitored.time-auxiliary.scram_start_time)
        controlled.power_CH2 = auxiliary.init_Power_Fraction_CH2*DecayHeatScalingFactor.powerCalculation(monitored.time-auxiliary.scram_start_time)
        controlled.power_CH3 = auxiliary.init_Power_Fraction_CH3*DecayHeatScalingFactor.powerCalculation(monitored.time-auxiliary.scram_start_time)
        #secondary system replaced by auxiliary secondary system
#    if monitored.time<(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux): # not yet auxiliary system up
        if controlled.high_pressure_secondary_A >= auxiliary.InitialOutletSecPress: 
            if (auxiliary.InitialOutletSecPress + (auxiliary.initialInletSecPress - auxiliary.InitialOutletSecPress)*(1-PumpCoastDownSec.flowrateCalculation(monitored.time-auxiliary.scram_start_time))) >= (auxiliary.InitialOutletSecPress+29.35152e+4):   
                 controlled.high_pressure_secondary_A = auxiliary.InitialOutletSecPress + (auxiliary.initialInletSecPress - auxiliary.InitialOutletSecPress)*(1-PumpCoastDownSec.flowrateCalculation(monitored.time-auxiliary.scram_start_time))                            
            else:
                 controlled.high_pressure_secondary_A = auxiliary.InitialOutletSecPress+29.35152e+3
        else:
              controlled.high_pressure_secondary_A = auxiliary.InitialOutletSecPress+29.35152e+3 #otherwise just compensate hydro pressure
        if controlled.high_pressure_secondary_B >= auxiliary.InitialOutletSecPress:
            if (auxiliary.InitialOutletSecPress + (auxiliary.initialInletSecPress - auxiliary.InitialOutletSecPress)*(1-PumpCoastDownSec.flowrateCalculation(monitored.time-auxiliary.scram_start_time))) >= (auxiliary.InitialOutletSecPress+29.35152e+4):   
                controlled.high_pressure_secondary_B = auxiliary.InitialOutletSecPress + (auxiliary.initialInletSecPress - auxiliary.InitialOutletSecPress)*(1-PumpCoastDownSec.flowrateCalculation(monitored.time-auxiliary.scram_start_time))                   
            else:
                 controlled.high_pressure_secondary_B = auxiliary.InitialOutletSecPress+29.35152e+3    
        else:
            controlled.high_pressure_secondary_B = auxiliary.InitialOutletSecPress+29.35152e+3 #otherwise just compensate hydro pressure              
#    else: # auxiliary system up
#        if controlled.high_pressure_secondary_A<15198299.45:                       #check if it has not already reached the final value
#            controlled.high_pressure_secondary_A = auxiliary.InitialOutletSecPress + (15198299.45-auxiliary.InitialOutletSecPress)/5*(monitored.time-(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux))
#        if controlled.high_pressure_secondary_B<15198299.45:                       #check if it has not already reached the final value
#            controlled.high_pressure_secondary_B = auxiliary.InitialOutletSecPress + (15198299.45-auxiliary.InitialOutletSecPress)/5*(monitored.time-(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux))

    if (monitored.max_temp_clad_CH1>auxiliary.CladTempTreshold) or (monitored.max_temp_clad_CH2>auxiliary.CladTempTreshold) or (monitored.max_temp_clad_CH3>auxiliary.CladTempTreshold):
        auxiliary.CladDamaged = True
        raise NameError ('exit condition reached - failure of the clad')
    return 

