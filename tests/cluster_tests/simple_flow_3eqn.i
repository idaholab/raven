[GlobalParams]
  # 2=2 eqn, 1D isothermal flow
  # 3=3 eqn, 1D non-isothermal flow
  # 7=7 eqn, 1D 2-phase flow
  model_type = 3
  gravity = '0 0 0'

  stabilization_type = 'NONE'

  scaling_factor_1phase = '1. 1.e-4 1.e-6'
[]
[FluidProperties]
  [./eos]
    type = LinearFluidProperties
    p_0 = 1.e5      # Pa
    rho_0 = 1.e3    # kg/m^3
    a2 = 1.e7       # m^2/s^2
    beta = -.46e-3  # K^{-1}  #Comment: This number should be positive for water. alpha = 1/V dV/dT = -1/rho d_rho/dT
    cv = 4.18e3     # J/kg-K, could be a global parameter?
    e_0 = 1.254e6   # J/kg
    T_0 = 300       # K
  [../]
[]
[Components]
  [./pipe]
    # geometry
    type = Pipe
    fp = eos
    position = '0 0 0'
    orientation = '1 0 0'
    A = 1.
    Dh = 1.12837916709551
    aw = 3.54490770181103
    f = 0.0
    Hw = 0.0 # convective heat transfer coefficient
    Tw = 310 # wall temperature
    length = 1
    n_elems = 1000
  [../]
  [./inlet]
    type = Inlet
    input = 'pipe(in)'
    p = 1e5
    T = 300.0
    fp = eos
  [../]
  [./outlet]
    type = Outlet
    input = 'pipe(out)'
    p = 9.5e4
    fp = eos
  [../]
[]

[Preconditioning]
  [./SMP_PJFNK]
    type = SMP
    full = true

    # Preconditioned JFNK (default)
    solve_type = 'PJFNK'

    petsc_options_iname = '-mat_mffd_type'
    petsc_options_value = 'ds'
  [../]
[]

[Executioner]
  # nl_abs_step_tol = 1e-15
  # close Executioner section
  type = RavenExecutioner
  control_logic_file = 'simple_flow_3eqn_control.py'
  dt = 1.e-4 # With pc_type=lu, we can use dt=1
  dtmin = 1.e-7
  petsc_options_iname = '-ksp_gmres_restart'
  petsc_options_value = 300
  nl_rel_tol = 1e-9
  nl_abs_tol = 1e-8
  nl_max_its = 30
  l_tol = 1e-8 # Relative linear tolerance for each Krylov solve
  l_max_its = 100 # Number of linear iterations for each Krylov solve
  start_time = 0.0
  num_steps = 10 # The number of timesteps in a transient run
  [./Quadrature]
    # Specify the order as FIRST, otherwise you will get warnings in DEBUG mode...
    type = TRAP
    order = FIRST
  [../]
[]

[Controlled]
  [./pipe_Area]
    print_csv = true
    component_name = pipe
    property_name = Area
    data_type = double
  [../]
  [./pipe_Dh]
    print_csv = true
    component_name = pipe
    property_name = Dh
    data_type = double
  [../]
  [./pipe_Hw]
    print_csv = true
    component_name = pipe
    property_name = Hw
    data_type = double
  [../]
  [./pipe_Tw]
    print_csv = true
    component_name = pipe
    property_name = Tw
    data_type = double
  [../]
  [./pipe_f]
    print_csv = true
    component_name = pipe
    property_name = f
    data_type = double
  [../]
[]
[Monitored]
[]
