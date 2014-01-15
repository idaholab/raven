import distribution1D
distcont  = distribution1D.DistributionContainer.Instance()


def initial_function(monitored, controlled, auxiliary):
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    mult = 1.01
    controlled.pipe1_Area = mult*controlled.pipe1_Area
    controlled.pipe1_Dh = mult*controlled.pipe1_Dh
    controlled.pipe1_Hw = mult*controlled.pipe1_Hw
    controlled.pipe1_aw = mult*controlled.pipe1_aw
    controlled.pipe1_f = mult*controlled.pipe1_f
    controlled.pipe2_Area = mult*controlled.pipe2_Area
    controlled.pipe2_Dh = mult*controlled.pipe2_Dh
    controlled.pipe2_Hw = mult*controlled.pipe2_Hw
    controlled.pipe2_aw = mult*controlled.pipe2_aw
    controlled.pipe2_f = mult*controlled.pipe2_f
    controlled.pump_mass_flow_rate = mult*controlled.pump_mass_flow_rate
    controlled.inlet_TDV_p_bc = mult*controlled.inlet_TDV_p_bc
    controlled.inlet_TDV_T_bc = mult*controlled.inlet_TDV_T_bc
    controlled.inlet_TDV_void_fraction_bc = mult*controlled.inlet_TDV_void_fraction_bc
    controlled.outlet_TDV_p_bc = mult*controlled.outlet_TDV_p_bc
    controlled.outlet_TDV_T_bc = mult*controlled.outlet_TDV_T_bc
    controlled.outlet_TDV_void_fraction_bc = mult*controlled.outlet_TDV_void_fraction_bc
    auxiliary.dummy_for_branch = 0.0
    random_n_1 = distcont.random()
    auxiliary.dummy_for_branch = distcont.randGen('zeroToOne',random_n_1)

    return

def control_function(monitored, controlled, auxiliary):
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    mult = 1.01
    controlled.pipe1_Area = mult*controlled.pipe1_Area
    controlled.pipe1_Dh = mult*controlled.pipe1_Dh
    controlled.pipe1_Hw = mult*controlled.pipe1_Hw
    controlled.pipe1_aw = mult*controlled.pipe1_aw
    controlled.pipe1_f = mult*controlled.pipe1_f
    controlled.pipe2_Area = mult*controlled.pipe2_Area
    controlled.pipe2_Dh = mult*controlled.pipe2_Dh
    controlled.pipe2_Hw = mult*controlled.pipe2_Hw
    controlled.pipe2_aw = mult*controlled.pipe2_aw
    controlled.pipe2_f = mult*controlled.pipe2_f
    controlled.pump_mass_flow_rate = mult*controlled.pump_mass_flow_rate
    controlled.inlet_TDV_p_bc = mult*controlled.inlet_TDV_p_bc
    controlled.inlet_TDV_T_bc = mult*controlled.inlet_TDV_T_bc
    controlled.inlet_TDV_void_fraction_bc = mult*controlled.inlet_TDV_void_fraction_bc
    controlled.outlet_TDV_p_bc = mult*controlled.outlet_TDV_p_bc
    controlled.outlet_TDV_T_bc = mult*controlled.outlet_TDV_T_bc
    controlled.outlet_TDV_void_fraction_bc = mult*controlled.outlet_TDV_void_fraction_bc
    return

