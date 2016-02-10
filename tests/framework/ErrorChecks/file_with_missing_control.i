[GlobalParams]
  # 2=2 eqn, 1D isothermal flow
  # 3=3 eqn, 1D non-isothermal flow
  # 7=7 eqn, 1D 2-phase flow
  initial_P = 1.e5
  initial_v = 0.0
  initial_T = 300.
  model_type = 3
  stabilization_type = SUPG
  scaling_factor_var = '1e4 1e1 1e-2'
[]
[FluidProperties]
  [./eos]
    type = LinearFluidProperties
    p_0 = 1.e5 # Pa
    rho_0 = 1.e3 # kg/m^3
    a2 = 1.e7 # m^2/s^2
    beta = .46e-3 # Comment: This number should be positive for water. alpha = 1/V dV/dT = -1/rho d_rho/dT
    cv = 4.18e3 # J/kg-K, could be a global parameter?
    e_0 = 1.254e6 # J/kg
    T_0 = 300 # K
  [../]
[]
[Components]
  # Pipes
  active = 'pipe1 pipe2 pump inlet_TDV outlet_TDV'
  [./pipe1]
    # geometry
    type = Pipe
    fp = eos
    position = '0 0 0'
    orientation = '1 0 0'
    A = 3.14159265E-04 # 2.0 cm (0.02 m) in diameter, A = 1/4 * PI * d^2
    Dh = 2.e-2
    f = 0.01
    Hw = 0.
    length = 1
    n_elems = 10
  [../]
  [./pipe2]
    # geometry
    type = Pipe
    fp = eos
    position = '1.2 0 0'
    orientation = '1 0 0'
    A = 0.785398163e-4 # 1.0 cm (0.01 m) in diameter, A = 1/4 * PI * d^2
    Dh = 1.e-2
    f = 0.01
    Hw = 0.
    length = 1
    n_elems = 10
  [../]
  [./pump]
    type = Pump
    fp = eos
    inputs = 'pipe1(out)'
    outputs = 'pipe2(in)'
    head = 1.0
    K_reverse = '10. 10.'
    A_ref = 0.785398163e-4
    initial_p = 1.e5
  [../]
  [./inlet_TDV]
    type = TimeDependentVolume
    input = 'pipe1(in)'
    p = 1.0e5
    T = 300.
    fp = eos
  [../]
  [./outlet_TDV]
    type = TimeDependentVolume
    input = 'pipe2(out)'
    p = 1.e5
    T = 300.
    fp = eos
  [../]
[]
[Preconditioning]
  # Uncomment one of the lines below to activate one of the blocks...
  # active = 'SMP_Newton'
  # active = 'FDP_PJFNK'
  # active = 'FDP_Newton'
  # The definitions of the above-named blocks follow.
  # End preconditioning block
  active = 'SMP_PJFNK'
  [./SMP_PJFNK]
    # Preconditioned JFNK (default)
    type = SMP
    full = true
    solve_type = PJFNK
    petsc_options_iname = '-mat_fd_type  -mat_mffd_type'
    petsc_options_value = 'ds ds'
    line_search = basic
  [../]
  [./SMP_Newton]
    type = SMP
    full = true
    solve_type = NEWTON
  [../]
  [./FDP_PJFNK]
    # Preconditioned JFNK (default)
    # petsc_options_iname = '-mat_fd_type'
    # petsc_options_value = 'ds'
    type = FDP
    full = true
    solve_type = PJFNK
    petsc_options_iname = '-mat_fd_coloring_err'
    petsc_options_value = 1.e-10
  [../]
  [./FDP_Newton]
    # petsc_options_iname = '-mat_fd_type'
    # petsc_options_value = 'ds'
    type = FDP
    full = true
    solve_type = NEWTON
    petsc_options_iname = '-mat_fd_coloring_err'
    petsc_options_value = 1.e-10
  [../]
[]
[Executioner]
  # These options *should* append to any options set in Preconditioning blocks above
  # petsc_options_iname = '-ksp_gmres_restart -pc_type'
  # petsc_options_value = '300 lu'
  # nl_abs_step_tol = 1e-15
  # close Executioner section
  type = RavenExecutioner
  control_logic_file = 'no_such_module.py'
  scheme = implicit-euler
  dt = 1.e-2
  dtmin = 1.e-5
  nl_rel_tol = 1e-8
  nl_abs_tol = 1e-8
  nl_max_its = 10
  l_tol = 1e-3 # Relative linear tolerance for each Krylov solve
  l_max_its = 100 # Number of linear iterations for each Krylov solve
  start_time = 0.0
  num_steps = 10 # The number of timesteps in a transient run
  [./Quadrature]
    # Specify the order as FIRST, otherwise you will get warnings in DEBUG mode...
    type = TRAP
    order = FIRST
  [../]
[]
[Outputs]
  [./out]
    type = Exodus
    use_displaced = true
    sequence = false
    append_displaced = true
  [../]
[]
[Controlled]
  [./pump_Head]
    component_name = pump
    property_name = head
    data_type = double
  [../]
  [./pump_inlet_K_reverse]
    component_name = pump
    property_name = 'inlet_K_reverse'
    data_type = double
  [../]
  [./pump_outlet_K_reverse]
    component_name = pump
    property_name = 'outlet_K_reverse'
    data_type = double
  [../]
  [./pump_Area]
    component_name = pump
    property_name = A_ref
    data_type = double
  [../]
  [./inlet_TDV_p_bc]
    component_name = 'inlet_TDV'
    property_name = 'p'
    data_type = double
  [../]
  [./inlet_TDV_T_bc]
    component_name = 'inlet_TDV'
    property_name = 'T'
    data_type = double
  [../]
  [./inlet_TDV_volume_fraction_vapor_bc]
    component_name = 'inlet_TDV'
    property_name = 'volume_fraction_vapor'
    data_type = double
  [../]
  [./outlet_TDV_p_bc]
    component_name = 'outlet_TDV'
    property_name = 'p'
    data_type = double
  [../]
  [./outlet_TDV_T_bc]
    component_name = 'outlet_TDV'
    property_name = 'T'
    data_type = double
  [../]
  [./outlet_TDV_volume_fraction_vapor_bc]
    component_name = 'outlet_TDV'
    property_name = 'volume_fraction_vapor'
    data_type = double
  [../]
[]
[Monitored]
[]
