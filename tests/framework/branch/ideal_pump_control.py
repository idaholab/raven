import distribution1D
distcont  = distribution1D.DistributionContainer.Instance()


def initial_function(monitored, controlled, auxiliary):
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    mult = 1.01
    controlled.pipe1_Area = mult*controlled.pipe1_Area
    controlled.pipe1_Hw = mult*controlled.pipe1_Hw
    controlled.pipe1_f = mult*controlled.pipe1_f
    controlled.pipe2_Area = mult*controlled.pipe2_Area
    controlled.pipe2_Hw = mult*controlled.pipe2_Hw
    controlled.pipe2_f = mult*controlled.pipe2_f
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
    print('THRESHOLDDDDDD ' + str(distcont.getVariable('ProbabilityThreshold','zeroToOne')))
    controlled.pipe1_Area = mult*controlled.pipe1_Area
    controlled.pipe1_Hw = mult*controlled.pipe1_Hw
    controlled.pipe1_f = mult*controlled.pipe1_f
    controlled.pipe2_Area = mult*controlled.pipe2_Area
    controlled.pipe2_Hw = mult*controlled.pipe2_Hw
    controlled.pipe2_f = mult*controlled.pipe2_f
    controlled.pump_mass_flow_rate = mult*controlled.pump_mass_flow_rate
    controlled.inlet_TDV_p_bc = mult*controlled.inlet_TDV_p_bc
    controlled.inlet_TDV_T_bc = mult*controlled.inlet_TDV_T_bc
    controlled.inlet_TDV_void_fraction_bc = mult*controlled.inlet_TDV_void_fraction_bc
    controlled.outlet_TDV_p_bc = mult*controlled.outlet_TDV_p_bc
    controlled.outlet_TDV_T_bc = mult*controlled.outlet_TDV_T_bc
    controlled.outlet_TDV_void_fraction_bc = mult*controlled.outlet_TDV_void_fraction_bc
    return

def dynamic_event_tree(monitored, controlled, auxiliary): 
  print("######################################## In dynamic_event_tree")
  if distcont.checkCdf('zeroToOne',auxiliary.dummy_for_branch) and (not auxiliary.aBoolean):
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    auxiliary.aBoolean = True
  return
