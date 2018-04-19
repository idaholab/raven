[outputParams]
  names = "dummy_for_branch pump_mass_flow_rate"
  start = "2.0 3.0"
  dt = "0.5 0.1"
  dx = "0.2 -0.3"
  trigger_low = "12.0 5.0"
  trigger_high = "12.1 5.1"
  start_time = 0.0
  end_time = 30.0
  x_start = 1.0
  x_dt = 0.1
  timesteps = 10
  time_delta = 1.0
  end_dx = "0.1 -0.1"
  end_probability = "0.6 0.4"
  trigger_name = zeroToOne
[]
