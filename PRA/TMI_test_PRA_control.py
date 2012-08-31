import sys
import math
import distribution1D
import raventools
distcont  = distribution1D.DistributionContainer.Instance()
DecayHeatScalingFactor     = raventools.decayHeat(1,1,3600*24*30*8,1)
PumpCoastDown              = raventools.pumpCoastdown(1,1)

def initial_function(monitored, controlled, auxiliary):

    
    auxiliary.scram_start_time               = 20     #it depends if we are restarting otherwise it should be the steady state reached point

#Status variable
    #plant
    auxiliary.ScramStatus                      = False
    #AuxSystem
    auxilary.AuxDieselAvailable                = True  # diesel generator available
    auxiliary.AuxDieselSuppliy                 = False # diesel generator not supplying
    auxiliary.AuxDieselRecoveryTime            = 0     
    #Primary Pumps
    auxiliary.PrimaryPump                      = 0     # 0 for steady 1 for cost down
    auxiliary.PrimaryPumpTransStart            = 0     # time at which the cost down start
    #Secondary Pumps 
    auxiliary.SecondaryPump                    = 0     # 0 for steady 1 for cost down 2 for up-rate
    auxiliary.SecondaryPumpTransStart          = 0     # time at which the cost down start
    #Grid
    auxiliary.PowerStatus                      = 0     # 0 steady, 1 decay heat
    #clad
    auxiliary.CladDamaged                      = False
    
#Nominal/initial values
    auxiliary.InitialHeadPrimary               = controlled.head_PumpB
    auxiliary.InitialPowerCH1                  = controlled.power_CH1
    auxiliary.InitialPowerCH2                  = controlled.power_CH2
    auxiliary.InitialPowerCH3                  = controlled.power_CH3
    auxiliary.initialInletSecPress             = controlled.high_pressure_secondary_A
    auxiliary.InitialOutletSecPress            = 151.7e5 #fix me
    
    
#Probability driven characteristics
    auxiliary.CladTempTreshold                 =  auxiliary.CladTempTreshold #*distribution.10_perc()
    if (distcont.random(0,1))>0.8                        # a 20% probability of not availability of diesel generators
        auxiliary.auxilary.AuxDieselAvailable  = False   # sorry not lucky
        distcont(distcont.random(0,1),2*3600)
    
     
    
    
    auxiliary.CladTemTreshold                =  auxiliary.CladTemTreshold #*distribution.10_perc()
    auxiliary.DeltaTimeScramToAux            = 3 #*distribution.10_perc()
    
    
    
    return

def control_function(monitored, controlled, auxiliary):
    
    if monitored.time>=auxiliary.scram_start_time: #we are in scram situation
#        
#        #primary pump B
        if controlled.head_PumpB>0.0001*auxiliary.InitialHeadPrimary:
            controlled.head_PumpB = auxiliary.InitialHeadPrimary*PumpCoastDown(monitored.time-auxiliary.scram_start_time) 
        else:
             controlled.head_PumpB = 0

        #primary pump A        
        if controlled.head_PumpA>0.0001*auxiliary.InitialHeadPrimary:
            controlled.head_PumpA = auxiliary.InitialHeadPrimary*PumpCoastDown(monitored.time-auxiliary.scram_start_time) 
        else:
             controlled.head_PumpA = 0
        
        #core power following decay heat curve       
        controlled.power_CH1 =DecayHeatScalingFactor(monitored.time-auxiliary.scram_start_time)*auxiliary.InitialPowerCH1
        controlled.power_CH2 =DecayHeatScalingFactor(monitored.time-auxiliary.scram_start_time)*auxiliary.InitialPowerCH2
        controlled.power_CH3 =DecayHeatScalingFactor(monitored.time-auxiliary.scram_start_time)*auxiliary.InitialPowerCH3

        #secondary system replaced by auxiliary secondary system
        if monitored.time<(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux): # not yet auxiliary system up
            if controlled.high_pressure_secondary_A >= auxiliary.InitialOutletSecPress: #just be sure not to be in reverse flow
                controlled.high_pressure_secondary_A = PumpCoastDown(monitored.time-auxiliary.scram_start_time) 
            else:
                controlled.high_pressure_secondary_A = auxiliary.InitialOutletSecPress #otherwise just compensate hydro pressure

            if controlled.high_pressure_secondary_B >= auxiliary.InitialOutletSecPress: #just be sure not to be in reverse flow
                controlled.high_pressure_secondary_B = PumpCoastDown(monitored.time-auxiliary.scram_start_time) 
            else:
                controlled.high_pressure_secondary_B = auxiliary.InitialOutletSecPress #otherwise just compensate hydro pressure
        else: # auxiliary system up
            if controlled.high_pressure_secondary_A<15198299.45:                       #check if it has not already reached the final value
                controlled.high_pressure_secondary_A = auxiliary.InitialOutletSecPress + (15198299.45-auxiliary.InitialOutletSecPress)/5*(monitored.time-(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux))
            if controlled.high_pressure_secondary_B<15198299.45:                       #check if it has not already reached the final value
                controlled.high_pressure_secondary_B = auxiliary.InitialOutletSecPress + (15198299.45-auxiliary.InitialOutletSecPress)/5*(monitored.time-(auxiliary.scram_start_time+auxiliary.DeltaTimeScramToAux))
        
    if (monitored.max_temp_clad_CH1>auxiliary.CladTemTreshold) or (monitored.max_temp_clad_CH2>auxiliary.CladTemTreshold) or (monitored.max_temp_clad_CH3>auxiliary.CladTemTreshold):
        auxiliary.CladDamaged = True
        raise NameError ('exit condition reached')

 
    return 

