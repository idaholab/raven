

def initial_function(monitored, controlled, auxiliary):
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    mult = 1.0
    controlled.pipe_Area = mult*controlled.pipe_Area
    controlled.pipe_Dh = mult*controlled.pipe_Dh
    controlled.pipe_Hw = mult*controlled.pipe_Hw
    controlled.pipe_Tw = mult*controlled.pipe_Tw
    controlled.pipe_aw = mult*controlled.pipe_aw
    controlled.pipe_f = mult*controlled.pipe_f
    auxiliary.Dummy1 = distributions.TestDistribution.getDistributionRandom()
    return

def control_function(monitored, controlled, auxiliary):
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    mult = 1.0
    if monitored.time_step > 3 and auxiliary.Dummy1 > 1.0:
      a = 1/0
    controlled.pipe_Area = mult*controlled.pipe_Area
    controlled.pipe_Dh = mult*controlled.pipe_Dh
    controlled.pipe_Hw = mult*controlled.pipe_Hw
    controlled.pipe_Tw = mult*controlled.pipe_Tw
    controlled.pipe_aw = mult*controlled.pipe_aw
    controlled.pipe_f = mult*controlled.pipe_f
    return

