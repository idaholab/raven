import distribution1D
distcont  = distribution1D.DistributionContainer.Instance()


def initial_function(monitored, controlled, auxiliary):
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    auxiliary.dummy_for_branch = 0.0
    # Preconditioner MonteCarlo (RAVEN way)
    controlled.pump_mass_flow_rate = controlled.pump_mass_flow_rate + controlled.pump_mass_flow_rate*distributions.testPreconditionerMonteCarloDist2.getDistributionRandom()*0.1
    # Preconditioner Grid (RAVEN way)
    controlled.inlet_TDV_p_bc = controlled.inlet_TDV_p_bc + controlled.inlet_TDV_p_bc*0.1*distributions.testPreconditionerGridDist2.getDistributionRandom()
    # Preconditioner LHS (RAVEN way)
    controlled.pipe2_f = controlled.pipe2_f + controlled.pipe2_f*0.1*distributions.testPreconditionerLHSDist2.getDistributionRandom()
    # Preconditioner MonteCarlo (Sampled Variable directly from RAVEN FRAMEWORK)
    controlled.inlet_TDV_T_bc = controlled.inlet_TDV_T_bc + controlled.inlet_TDV_T_bc*0.1*auxiliary.testMCpreconditioned1
    # Preconditioner Grid (Sampled Variable directly from RAVEN FRAMEWORK)
    controlled.outlet_TDV_p_bc = controlled.outlet_TDV_p_bc + controlled.outlet_TDV_p_bc*0.1*auxiliary.testGridpreconditioned1
    # Preconditioner LHS (Sampled Variable directly from RAVEN FRAMEWORK)
    controlled.outlet_TDV_T_bc = controlled.outlet_TDV_T_bc + controlled.outlet_TDV_T_bc*0.1*auxiliary.testLHSpreconditioned1
    return

def control_function(monitored, controlled, auxiliary):
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    if auxiliary.dummy_for_branch < 1.0: auxiliary.dummy_for_branch = auxiliary.dummy_for_branch + 0.25
    print('THRESHOLDDDDDD ' + str(distributions.zeroToOne.getVariable('ProbabilityThreshold')))
    return

def dynamic_event_tree(monitored, controlled, auxiliary):
  if distcont.checkCdf('zeroToOne',auxiliary.dummy_for_branch) and (not auxiliary.aBoolean) and monitored.time_step>1:
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    auxiliary.aBoolean = True
  return
