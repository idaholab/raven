
def initial_function(monitored, controlled, auxiliary):
    #print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    auxiliary.Dummy1 = distributions.ANormalDistribution.getDistributionRandom()
    mult = 1.0
    controlled.pipe_Area = mult*controlled.pipe_Area
    controlled.pipe_Hw = mult*controlled.pipe_Hw
    controlled.pipe_Tw = mult*controlled.pipe_Tw
    return

def control_function(monitored, controlled, auxiliary):
    #print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    mult = 1.0
    controlled.pipe_Area = mult*controlled.pipe_Area
    controlled.pipe_Hw = mult*controlled.pipe_Hw
    controlled.pipe_Tw = mult*controlled.pipe_Tw
    return

