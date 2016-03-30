[GlobalParams]
  gravity = '0 0 0'

  initial_p = 0.1e6
  initial_v = 0
  initial_T = 300

  stabilization_type = 'NONE'
  scaling_factor_1phase = '1. 1. 1.'
[]

[FluidProperties]
  [./eos]
    type = IAPWS95LiquidFluidProperties
  [../]
[]

[Components]
  [./pipe]
    type = Pipe
    # geometry
    position = '0 0 0'
    orientation = '1 0 0'
    length = 1
    n_elems = 100

    A = 1.907720E-04
    f = 0.0
    Tw = 310
    Hw = 0.0

    fp = eos
  [../]

  [./inlet]
    type = Inlet
    input = 'pipe(in)'

    rho = 996.556340388366266
    u = 2
  [../]

  [./outlet]
    type = Outlet
    input = 'pipe(out)'

    p = 0.1e6
  [../]
[]



[Preconditioning]
  [./SMP_PJFNK]
    type = SMP
    full = true
    solve_type = 'PJFNK'
  [../]
[]

[Executioner]
  type = RavenExecutioner
  control_logic_file = 'clg_massflowrate_3eqn_control.py'
  legacy = false
  scheme = 'bdf2'

  dt = 1.e-1
  dtmin = 1.e-7

  nl_rel_tol = 1e-9
  nl_abs_tol = 1e-8
  nl_max_its = 30

  l_tol = 1e-3
  l_max_its = 100

  start_time = 0.0
  num_steps = 1 # The number of timesteps in a transient run

  [./Quadrature]
    type = GAUSS
    order = THIRD
  [../]
[]

[Controlled]
  [./pipe_Area]
    print_csv = true
    component_name = pipe
    property_name = Area
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
[]
[Monitored]
[]
[Distributions]
  [./ANormalDistribution]
    mu = -4
    type = NormalDistribution
    sigma = 2
  [../]
[]
[Auxiliary]
  [./Dummy1]
    print_csv = true
    data_type = double
    initial_value = 0.0
  [../]
[]
