import sys
import math
import distribution1D
import crowtools
# initialize distribution container

def keep_going_function(monitored, controlled, auxiliary): return auxiliary.keepGoing

def restart_function(monitored, controlled, auxiliary):
    # here we store some critical parameters that we want in the output
    auxiliary.CladFailureDistThreshold = distributions.CladFailureDist.getVariable('ProbabilityThreshold')
    auxiliary.CladTempBranched = distributions.CladFailureDist.inverseCdf(auxiliary.CladFailureDistThreshold)
    
    auxiliary.crew1DG1Threshold = distributions.crew1DG1.getVariable('ProbabilityThreshold')
    auxiliary.DG1recoveryTime   = distributions.crew1DG1.inverseCdf(auxiliary.crew1DG1Threshold)

    auxiliary.PrimPGrecoveryThreshold = distributions.PrimPGrecovery.getVariable('ProbabilityThreshold')
    auxiliary.PrimPGrecoveryTime = distributions.PrimPGrecovery.inverseCdf(auxiliary.PrimPGrecoveryThreshold)

    auxiliary.crewSecPGThreshold = distributions.crewSecPG.getVariable('ProbabilityThreshold')
    auxiliary.SecPGrecoveryTime  = distributions.crewSecPG.inverseCdf(auxiliary.crewSecPGThreshold)

    auxiliary.deltaAux = min(auxiliary.DG1recoveryTime,auxiliary.SecPGrecoveryTime,auxiliary.PrimPGrecoveryTime)

    # here we check the variables one by one (for the aux)
    return

def control_function(monitored, controlled, auxiliary):
    if auxiliary.CladDamaged and not auxiliary.AuxSystemUp:  auxiliary.keepGoing = False
    # here we check the variables one by one (for the aux)
    if monitored.time>=auxiliary.scram_start_time:
        auxiliary.ScramStatus = True
        print('SCRAM')
    else:
        auxiliary.ScramStatus = False
        print('OPERATIONAL STATE')

    if (auxiliary.crew1DG1) and not auxiliary.AuxSystemUp:
        auxiliary.AuxSystemUp =  True
    if (auxiliary.PrimPGrecovery) and not auxiliary.AuxSystemUp:
        auxiliary.AuxSystemUp =  True
    if (auxiliary.crewSecPG) and not auxiliary.AuxSystemUp:
        auxiliary.AuxSystemUp =  True

    #if auxiliary.CladDamaged:
    #        if monitored.time_step > 1:
    #            raise NameError ('exit condition reached - failure of the clad')
    if auxiliary.ScramStatus: #we are in scram
        #primary pump B
        if controlled.Head_PumpB>1.e-4*8.9:
            if not auxiliary.AuxSystemUp: # not yet auxiliary system up
                controlled.Head_PumpB = tools.PumpCoastDown.compute(monitored.time-auxiliary.scram_start_time)
                if controlled.Head_PumpB < (1.e-4*8.9): controlled.Head_PumpB = 1.e-4*8.9
                controlled.friction1_SC_B = auxiliary.frict_m*controlled.Head_PumpB + auxiliary.frict_q
                auxiliary.init_exp_frict = True
            else: #system up
                controlled.friction1_SC_B = controlled.friction1_SC_B - 62.5
                if controlled.friction1_SC_B < 0.1: controlled.friction1_SC_B = 0.1
                if controlled.Head_PumpB <= 0.10*8.9:
                    controlled.Head_PumpB = controlled.Head_PumpB + 0.00125*8.9
                    if controlled.Head_PumpB > 0.10*8.9: controlled.Head_PumpB = 0.10*8.9
                else:
                    controlled.Head_PumpB = tools.PumpCoastDown.compute(monitored.time-auxiliary.scram_start_time)
                    if controlled.Head_PumpB < (0.10*8.9):
                        controlled.Head_PumpB = 0.10*8.9 #
        else: # auxiliary.Head_PumpB<1.e-4*8.9
            if not auxiliary.AuxSystemUp: # not yet auxiliary system up
                controlled.Head_PumpB     = 1.e-4*8.9
                controlled.friction1_SC_B = 10000
            else: # auxiliary system up
                controlled.friction1_SC_B = controlled.friction1_SC_B - 62.5
                if controlled.friction1_SC_B < 0.1: controlled.friction1_SC_B = 0.1
                if controlled.Head_PumpB <= 0.10*8.9:
                    controlled.Head_PumpB = controlled.Head_PumpB + 0.00125*8.9
                    if controlled.Head_PumpB > 0.10*8.9: controlled.Head_PumpB = 0.10*8.9
                else:
                    controlled.Head_PumpB = tools.PumpCoastDown.compute(monitored.time-auxiliary.scram_start_time)
                    if controlled.Head_PumpB < (0.10*8.9):
                        controlled.Head_PumpB = 0.10*8.9

        #primary pump A
        controlled.Head_PumpA      = controlled.Head_PumpB
        controlled.friction2_SC_B = controlled.friction1_SC_B
        controlled.friction1_CL_B = controlled.friction1_SC_B
        controlled.friction2_CL_B = controlled.friction1_SC_B
        controlled.friction1_SC_A = controlled.friction1_SC_B
        controlled.friction2_SC_A = controlled.friction1_SC_B
        controlled.friction1_CL_A = controlled.friction1_SC_B
        controlled.friction2_CL_A = controlled.friction1_SC_B
    print(controlled.friction1_SC_B)
    #secondary system replaced by auxiliary secondary system
    if not auxiliary.AuxSystemUp and auxiliary.ScramStatus: # not yet auxiliary system up
        print('not yet auxiliary system up')
        controlled.MassFlowRateIn_SC_B = 4.542*tools.PumpCoastDownSec.compute(monitored.time-auxiliary.scram_start_time)
        controlled.MassFlowRateIn_SC_A = 4.542*tools.PumpCoastDownSec.compute(monitored.time-auxiliary.scram_start_time)
        if controlled.MassFlowRateIn_SC_A < (1.e-4*4.542):
            controlled.MassFlowRateIn_SC_A = 1.e-4*4.542
            controlled.MassFlowRateIn_SC_B = 1.e-4*4.542
    if auxiliary.AuxSystemUp and auxiliary.ScramStatus: # auxiliary system up
        print('auxiliary system up')
        controlled.MassFlowRateIn_SC_B = 4.542*0.10
        if controlled.MassFlowRateIn_SC_B <= 0.10*4.542:
            controlled.MassFlowRateIn_SC_B = 0.10*4.542
        else:
            controlled.MassFlowRateIn_SC_B = 4.542*tools.PumpCoastDownSec.compute(monitored.time-auxiliary.scram_start_time)
            if controlled.MassFlowRateIn_SC_B <= 0.10*4.542:
                controlled.MassFlowRateIn_SC_B = 0.10*4.542
        controlled.MassFlowRateIn_SC_A = controlled.MassFlowRateIn_SC_B
    if auxiliary.ScramStatus:
        #core power following decay heat curve
        controlled.power_CH1 = auxiliary.init_Power_Fraction_CH1*tools.DecayHeatScalingFactor.compute(monitored.time-auxiliary.scram_start_time)
        controlled.power_CH2 = auxiliary.init_Power_Fraction_CH2*tools.DecayHeatScalingFactor.compute(monitored.time-auxiliary.scram_start_time)
        controlled.power_CH3 = auxiliary.init_Power_Fraction_CH3*tools.DecayHeatScalingFactor.compute(monitored.time-auxiliary.scram_start_time)

    return

def dynamic_event_tree(monitored, controlled, auxiliary):
    if monitored.time_step <= 1: return
    if distributions.CladFailureDist.checkCdf(monitored.avg_temp_clad_CH1) and (not auxiliary.CladDamaged) and (not auxiliary.AuxSystemUp):
        auxiliary.CladDamaged = True
        return
    if distributions.CladFailureDist.checkCdf(monitored.avg_temp_clad_CH2) and (not auxiliary.CladDamaged) and (not auxiliary.AuxSystemUp):
        auxiliary.CladDamaged = True
        return
    if distributions.CladFailureDist.checkCdf(monitored.avg_temp_clad_CH3) and (not auxiliary.CladDamaged) and (not auxiliary.AuxSystemUp):
        auxiliary.CladDamaged = True
        return
    if distributions.crew1DG1.checkCdf(monitored.time - auxiliary.scram_start_time ) and (not auxiliary.CladDamaged) and (not auxiliary.crew1DG1) and (not auxiliary.AuxSystemUp):
        auxiliary.crew1DG1 = True
        return
    if distributions.PrimPGrecovery.checkCdf(monitored.time - auxiliary.scram_start_time) and (not auxiliary.CladDamaged) and (not auxiliary.PrimPGrecovery) and (not auxiliary.AuxSystemUp):
        auxiliary.PrimPGrecovery = True
        return
    if distributions.crewSecPG.checkCdf(monitored.time - auxiliary.scram_start_time) and (not auxiliary.CladDamaged) and (not auxiliary.crewSecPG) and (not auxiliary.AuxSystemUp):
        auxiliary.crewSecPG = True
        return
    return
