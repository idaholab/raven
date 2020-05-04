[outputParams]
  names = "dummy_for_branch pump_mass_flow_rate outlet_TDV_T_bc inlet_TDV_T_bc inlet_TDV_p_bc outlet_TDV_p_bc"
  start = "2.0 3.0 300.0 120.0 40.0 50.0"
  dt = "0.5 0.1 -0.1 0.2 0.3 0.3"
  dx = "0.2 -0.3 0.0 0.0 -0.2 -0.2"
  trigger_low = "12.0 5.0 290.0 150.0 50.0 60.0"
  trigger_high = "12.1 5.1 291.0 151.0 51.0 61.0"
  start_time = 0.0
  end_time = 30.0
  x_start = 1.0
  x_dt = 0.1
  timesteps = 10
  time_delta = 2.0
  end_dx = "0.1 -0.1"
  end_probability = "0.6 0.4"
  trigger_name = zeroToOne
[]
