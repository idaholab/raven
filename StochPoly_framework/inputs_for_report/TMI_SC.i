[GlobalParams]
  # 2=2 eqn, 1D isothermal flow
  # 3=3 eqn, 1D non-isothermal flow
  # 7=7 eqn, 1D 2-phase flow
  # scaling_factor_var = '1. 1.e-6 1.e-7'
  # supg = false
  model_type = 3
  global_init_P = 15.17e6
  global_init_V = 0.
  global_init_T = 564.15
  scaling_factor_var = '1.e-1 1.e-5 1.e-8'
[]

[EoS]
  [./eos]
    e_0 = 3290122.80 # J/kg
    beta = .46e-3 # K^{-1}
    a2 = 1.e7 # m^2/s^2
    rho_0 = 738.350 # kg/m^3
    T_0 = 564.15 # K
    type = NonIsothermalEquationOfState
    cv = 5.832e3 # J/kg-K
    p_0 = 15.17e6 # Pa
  [../]
[]

[Materials]
  [./fuel-mat]
    k = 3.65
    Cp = 288.734
    type = SolidMaterialProperties
    rho = 1.032e4
  [../]
  [./gap-mat]
    k = 1.084498
    Cp = 1.0
    type = SolidMaterialProperties
    rho = 1.
  [../]
  [./clad-mat]
    k = 16.48672
    Cp = 321.384
    type = SolidMaterialProperties
    rho = 6.55e3
  [../]
  [./wall-mat]
    k = 1.0
    Cp = 4.0
    type = SolidMaterialProperties
    rho = 80.0
  [../]
[]

[Components]
  # Core region components
  # [./high_pressure_seconday_A]
  # T_bc = 537.15
  # p_bc = '152.19e5'
  # eos = eos
  # input = 'pipe1-SC-A(in)'
  # type = TimeDependentVolume
  # [../]
  # [./high_pressure_seconday_B]
  # T_bc = 537.15
  # p_bc = '152.19e5'
  # eos = eos
  # input = 'pipe1-SC-B(in)'
  # type = TimeDependentVolume
  # [../]
  [./reactor]
    initial_power = 2.77199979e9
    type = Reactor
  [../]
  [./CH1]
    # peak_power = '6.127004e8 0. 0.'
    elem_number_of_hs = '3 1 1'
    Ts_init = 564.15
    orientation = '0 0 1'
    rho_hs = '1.0412e2 1.0 6.6e1'
    aw = 276.5737513
    n_elems = 8
    k_hs = '3.65 1.084498 16.48672'
    material_hs = 'fuel-mat gap-mat clad-mat'
    Dh = 0.01332254
    fuel_type = cylinder
    name_of_hs = 'FUEL GAP CLAD'
    Hw = 5.33e4
    n_heatstruct = 3
    A = 1.161864
    power_fraction = '3.33672612e-1 0 0'
    f = 0.01
    type = CoreChannel
    Cp_hs = '288.734 1.0 321.384'
    eos = eos
    length = 3.6576
    position = '0 -1.2 0'
    width_of_hs = '0.0046955  0.0000955  0.000673'
  [../]
  [./CH2]
    # peak_power = '5.094461e8 0. 0.'
    elem_number_of_hs = '3 1 1'
    Ts_init = 564.15
    orientation = '0 0 1'
    rho_hs = '1.0412e2 1. 6.6e1'
    aw = 276.5737513
    n_elems = 8
    k_hs = '3.65  1.084498  16.48672'
    material_hs = 'fuel-mat gap-mat clad-mat'
    Dh = 0.01332254
    fuel_type = cylinder
    name_of_hs = 'FUEL GAP CLAD'
    Hw = 5.33e4
    n_heatstruct = 3
    A = 1.549152542
    power_fraction = '3.69921461e-1 0 0'
    f = 0.01
    type = CoreChannel
    Cp_hs = '288.734  1.0  321.384'
    eos = eos
    length = 3.6576
    position = '0 0 0'
    width_of_hs = '0.0046955  0.0000955  0.000673'
  [../]
  [./CH3]
    # peak_power = '3.401687e8 0. 0.'
    elem_number_of_hs = '3 1 1'
    Ts_init = 564.15
    orientation = '0 0 1'
    rho_hs = '1.0412e2  1.0  6.6e1'
    aw = 276.5737513
    n_elems = 8
    k_hs = '3.65  1.084498  16.48672'
    material_hs = 'fuel-mat gap-mat clad-mat'
    Dh = 0.01332254
    fuel_type = cylinder
    name_of_hs = 'FUEL GAP CLAD'
    Hw = 5.33e4
    n_heatstruct = 3
    A = 1.858983051
    power_fraction = '2.96405926e-1 0 0'
    f = 0.01
    type = CoreChannel
    Cp_hs = '288.734  1.0  6.6e3'
    eos = eos
    length = 3.6576
    position = '0 1.2 0'
    width_of_hs = '0.0046955  0.0000955  0.000673'
  [../]
  [./bypass_pipe]
    A = 1.589571014
    orientation = '0 0 1'
    Dh = 1.42264
    f = 0.001
    Hw = 0.0
    eos = eos
    length = 3.6576
    n_elems = 5
    position = '0 1.5 0'
    type = Pipe
  [../]
  [./LowerPlenum]
    inputs =  'DownComer-A(out) DownComer-B(out)'
    Area =  3.618573408
    outputs =  'CH1(in) CH2(in) CH3(in) bypass_pipe(in)'
    K =  '0.2 0.2 0.2 0.2 0.4 40.0'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./UpperPlenum]
    inputs =  'CH1(out) CH2(out) CH3(out) bypass_pipe(out)'
    Area =  7.562307456
    outputs =  'pipe1-HL-A(in) pipe1-HL-B(in)'
    K =  '0.5 0.5 0.5 80.0 0.5 0.5'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./DownComer-A]
    A = 3.6185734
    orientation = '0 0 -1'
    Dh = 1.74724302
    f = 0.001
    Hw = 0.
    eos = eos
    length = 4
    n_elems = 3
    position = '0 2.0 4.0'
    type = Pipe
  [../]
  [./pipe1-HL-A]
    A = 7.562307456
    orientation = '0 0 1'
    Dh = 3.103003207
    f = 0.001
    Hw = 0.0
    eos = eos
    length = 4.
    n_elems = 3
    position = '0 0.5 4.0'
    type = Pipe
  [../]
  [./pipe2-HL-A]
    A = 2.624474
    orientation = '0 1 0'
    Dh = 1.828
    f = 0.001
    Hw = 0.0
    eos = eos
    length = 3.5
    n_elems = 3
    position = '0 0.5 8.0'
    type = Pipe
  [../]
  [./pipe1-CL-A]
    A = 2.624474
    orientation = '0 -1 0'
    Dh = 1.828
    f = 0.001
    Hw = 0.0
    eos = eos
    length = 1.
    n_elems = 3
    position = '0 3.0 4.0'
    type = Pipe
  [../]
  [./pipe2-CL-A]
    A = 2.624474
    orientation = '0 -1 0'
    Dh = 1.828
    f = 0.001
    Hw = 0.0
    eos = eos
    length = 0.8
    n_elems = 3
    position = '0 4 4.0'
    type = Pipe
  [../]
  [./pipe1-SC-A]
    A = 1.3122 # 2.624474
    orientation = '0 -1 0'
    Dh = 0.914 # 1.828
    f = 0.001
    Hw = 0.0
    eos = eos
    length = 1.
    n_elems = 3
    position = '0 5.2 4.0'
    type = Pipe
  [../]
  [./pipe2-SC-A]
    A = 1.3122 # 2.624474
    orientation = '0 1 0'
    Dh = 0.914
    f = 0.001
    Hw = 0.0
    eos = eos
    length = 1.
    n_elems = 3
    position = '0 4.2 8.0'
    type = Pipe
  [../]
  [./Branch1-A]
    inputs =  'pipe1-HL-A(out)'
    Area =  7.562307456
    outputs =  'pipe2-HL-A(in) pipe-to-Pressurizer(in)'
    K =  '0.5 0.7 80.'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./Branch2-A]
    inputs =  'pipe1-CL-A(out)'
    Area =  3.6185734
    outputs =  'DownComer-A(in)'
    K =  '0.5 0.7'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./Branch3-A]
    inputs =  'pipe2-HL-A(out)'
    Area =  2.624474
    outputs =  'HX-A(primary_in)'
    K =  '0.5 0.7'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./Pump-A]
    inputs = pipe2-CL-A(out)
    Head = 8.9
    Area = 2.624474
    outputs = pipe1-CL-A(in)
    eos = eos
    Initial_pressure = 151.7e5
    K_reverse = '2000 2000'
    type = Pump
  [../]
  [./HX-A]
    orientation = '0 0 -1'
    aw = 539.02
    n_elems = 10
    A_secondary = 5 # 5
    material_wall = wall-mat
    wall_thickness = 0.001
    Dh = 0.01
    Twall_init = 564.15
    Hw = 1.e4
    aw_secondary = 539.02 # 539.02
    eos_secondary = eos
    type = HeatExchanger
    A = 5.0
    Dh_secondary = 0.001
    Hw_secondary = 1.e4
    n_wall_elems = 2
    Cp_wall = 100.0
    f_secondary = 0.01
    rho_wall = 100.0
    k_wall = 100.0
    f = 0.01
    eos = eos
    length = 4.
    position = '0 4. 8.'
    dim_wall = 1
  [../]
  [./Branch4-A]
    inputs =  pipe1-SC-A(out)
    Area =  2.624474e2
    outputs =  HX-A(secondary_in)
    K =  '0.5 0.7'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./Branch5-A]
    inputs =  'HX-A(secondary_out)'
    Area =  2.624474e2
    outputs =  'pipe2-SC-A(in)'
    K =  '0.5 0.7'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./Branch6-A]
    inputs =  'HX-A(primary_out)'
    Area =  2.624474e2
    outputs =  'pipe2-CL-A(in)'
    K =  '0.5 0.7'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./PressureOutlet-SC-A]
    eos = eos
    input = pipe2-SC-A(out)
    p_bc = 151.7e5
    type = TimeDependentVolume
    T_bc = 564.15
  [../]
  [./DownComer-B]
    A = 3.6185734
    orientation = '0 0 -1'
    Dh = 1.74724302
    f = 0.001
    Hw = 0.
    eos = eos
    length = 4
    n_elems = 3
    position = '0 -2.0 4.0'
    type = Pipe
  [../]
  [./pipe1-HL-B]
    A = 7.562307456
    orientation = '0 0 1'
    Dh = 3.103003207
    f = 0.001
    Hw = 0.0
    eos = eos
    length = 4.
    n_elems = 3
    position = '0 -0.5 4.0'
    type = Pipe
  [../]
  [./pipe2-HL-B]
    A = 2.624474
    orientation = '0 -1 0'
    Dh = 1.828
    f = 0.001
    Hw = 0.0
    eos = eos
    length = 3.5
    n_elems = 3
    position = '0 -0.5 8.0'
    type = Pipe
  [../]
  [./pipe1-CL-B]
    A = 2.624474
    orientation = '0 1 0'
    Dh = 1.828
    f = 0.001
    Hw = 0.0
    eos = eos
    length = 1.
    n_elems = 3
    position = '0 -3.0 4.0'
    type = Pipe
  [../]
  [./pipe2-CL-B]
    A = 2.624474
    orientation = '0 1 0'
    Dh = 1.828
    f = 0.001
    Hw = 0.0
    eos = eos
    length = 0.8
    n_elems = 3
    position = '0 -4.0 4.0'
    type = Pipe
  [../]
  [./pipe1-SC-B]
    A = 1.3122 # 2.624474
    orientation = '0 1 0'
    Dh = 0.914 # 1.828
    f = 0.001
    Hw = 0.0
    eos = eos
    length = 1.
    n_elems = 3
    position = '0 -5.2 4.0'
    type = Pipe
  [../]
  [./pipe2-SC-B]
    A = 1.3122 # 2.624474
    orientation = '0 -1 0'
    Dh = 0.914 # 1.828
    f = 0.001
    Hw = 0.0
    eos = eos
    length = 1.
    n_elems = 3
    position = '0 -4.2 8.0'
    type = Pipe
  [../]
  [./Branch1-B]
    inputs =  'pipe1-HL-B(out)'
    Area =  7.562307456
    outputs =  'pipe2-HL-B(in)'
    K =  '0.5 0.7'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./Branch2-B]
    inputs =  'pipe1-CL-B(out)'
    Area =  3.6185734
    outputs =  'DownComer-B(in)'
    K =  '0.5 0.7'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./Branch3-B]
    inputs =  'pipe2-HL-B(out)'
    Area =  2.624474
    outputs =  'HX-B(primary_in)'
    K =  '0.5 0.7'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./Pump-B]
    inputs = pipe2-CL-B(out)
    Head = 8.9
    Area = 2.624474
    outputs = pipe1-CL-B(in)
    eos = eos
    Initial_pressure = 151.7e5
    K_reverse = '2000 2000'
    type = Pump
  [../]
  [./HX-B]
    orientation = '0 0 -1'
    aw = 539.02
    n_elems = 10
    A_secondary = 5 # 5.
    material_wall = wall-mat
    wall_thickness = 0.001
    Dh = 0.01
    Twall_init = 564.15
    Hw = 1.e4
    aw_secondary = 539.02 # 539.02
    eos_secondary = eos
    type = HeatExchanger
    A = 5.
    Dh_secondary = 0.001
    Hw_secondary = 1.e4
    n_wall_elems = 2
    Cp_wall = 100.0
    f_secondary = 0.01
    rho_wall = 100.0
    k_wall = 100.0
    f = 0.01
    eos = eos
    length = 4.
    position = '0 -4. 8.'
    disp_mode = -1.0
  [../]
  [./Branch4-B]
    inputs =  'pipe1-SC-B(out)'
    Area =  2.624474e2
    outputs =  'HX-B(secondary_in)'
    K =  '0.5 0.7'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./Branch5-B]
    inputs =  'HX-B(secondary_out)'
    Area =  2.624474e2
    outputs =  'pipe2-SC-B(in)'
    K =  '0.5 0.7'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./Branch6-B]
    inputs =  'HX-B(primary_out)'
    Area =  2.624474e2
    outputs =  'pipe2-CL-B(in)'
    K =  '0.5 0.7'
    eos =  eos
    Initial_pressure =  151.7e5
    type =  Branch
  [../]
  [./PressureOutlet-SC-B]
    eos = eos
    input = pipe2-SC-B(out)
    p_bc = 151.7e5
    type = TimeDependentVolume
    T_bc = 564.15
  [../]
  [./pipe-to-Pressurizer]
    A = 2.624474
    orientation = '0 0 1'
    Dh = 1.828
    f = 10.
    Hw = 0.0
    eos = eos
    length = 0.5
    n_elems = 3
    position = '0 0.5 8.0'
    type = Pipe
  [../]
  [./Pressurizer]
    eos = eos
    input = pipe-to-Pressurizer(out)
    p_bc = 151.7e5
    type = TimeDependentVolume
    T_bc = 564.15
  [../]
  [./MassFlowRateIn-SC-B]
    # type = TDM
    # massflowrate_bc = 8801.1
    v_bc = 2.542 # 4.542
    input = pipe1-SC-B(in)
    type = TimeDependentJunction
    eos = eos
    T_bc = 537.15
  [../]
  [./MassFlowRateIn-SC-A]
    # type = TDM
    # massflowrate_bc = 8801.1
    v_bc = 2.542 # 4.542
    input = pipe1-SC-A(in)
    type = TimeDependentJunction
    eos = eos
    T_bc = 537.15
  [../]
[]

[Preconditioning]
  # active = 'FDP_Newton'
  # End preconditioning block
  active = 'SMP_PJFNK'
  [./SMP_PJFNK]
    petsc_options_iname = '-mat_fd_type  -mat_mffd_type'
    full = true
    type = SMP
    petsc_options_value = 'ds             ds'
    petsc_options = -snes_mf_operator
  [../]
  [./SMP]
    full = true
    type = SMP
    petsc_options = -snes_mf_operator
  [../]
  [./FDP_PJFNK]
    # These options **together** cause a zero pivot in this problem, even without SUPG terms.
    # But using either option alone appears to be OK.
    # petsc_options_iname = '-mat_fd_coloring_err -mat_fd_type'
    # petsc_options_value = '1.e-10               ds'
    petsc_options_iname = -mat_fd_type
    full = true
    type = FDP
    petsc_options_value = ds
    petsc_options = '-snes_mf_operator -pc_factor_shift_nonzero'
  [../]
  [./FDP_Newton]
    # These options **together** cause a zero pivot in this problem, even without SUPG terms.
    # But using either option alone appears to be OK.
    # petsc_options_iname = '-mat_fd_coloring_err -mat_fd_type'
    # petsc_options_value = '1.e-10               ds'
    petsc_options_iname = -mat_fd_type
    full = true
    type = FDP
    petsc_options_value = ds
    petsc_options = -snes
  [../]
[]

[Executioner]
  # petsc_options_iname = '-ksp_gmres_restart -pc_type'
  # '300'
  # num_steps = '3'
  # time_t =  ' 0      1.0        3.0         5.01       9.5       9.75    14          17        60       61.1     100.8    101.5  102.2 120.0  400 1000 1.0e5'
  # time_dt =  '1.e-1  0.1        0.15        0.20       0.25      0.30    0.35        0.40    0.45      0.09      0.005     0.008   0.2   0.2    0.2 0.3  0.6'
  nl_abs_tol = 1e-8
  restart_file_base = TMI_test_PRA_transient_less_w_ss_out_restart_0831
  nl_rel_tol = 1e-5
  ss_check_tol = 1e-05
  perf_log = true
  nl_max_its = 120
  type = RavenExecutioner
  control_logic_file = TMI_PRA_trans_SC_control.py
  max_increase = 3
  petsc_options_value = lu # '300'
  l_max_its = 100 # Number of linear iterations for each Krylov solve
  start_time = 100.0
  predictor_scale = 0.6
  dtmax = 9999
  nl_rel_step_tol = 1e-3
  dt = 5e-5
  petsc_options_iname = -pc_type
  e_tol = 10.0
  l_tol = 1e-5 # Relative linear tolerance for each Krylov solve
  end_time = 2500.0
  e_max = 99999.
  [./TimeStepper]
    type = FunctionDT
    time_t = ' 0      1.0        3.0         5.01       9.5       9.75    14          17        60       61.1     100.8    101.5  102.2 120.0  2501.23 1.0e5'
    time_dt = '1.e-1  0.1        0.15        0.20       0.25      0.30    0.35        0.40    0.45      0.09      0.1      0.008   0.2   0.21   0.2  0.6'
  [../]
  [./Quadrature]
    type = TRAP
    order = FIRST
  [../]
[]

[Output]
  # xda = true
  # num_restart_files = 1
  output_initial = true
  output_displaced = false
  exodus = false
  file_base = TMI_test_PRA_transient_less_w_out
  postprocessor_csv = true
  max_pps_rows_screen = 25
[]

[Controlled]
  # control logic file name
  # [./high_pressure_secondary_A]
  # property_name = p_bc
  # print_csv = true
  # data_type = double
  # component_name = high_pressure_seconday_A
  # [../]
  # [./high_pressure_secondary_B]
  # property_name = p_bc
  # print_csv = true
  # data_type = double
  # component_name = high_pressure_seconday_B
  # [../]
  [./power_CH1]
    print_csv =  true
    data_type =  double
    property_name =  FUEL:power_fraction
    component_name =  CH1
  [../]
  [./power_CH2]
    print_csv =  true
    data_type =  double
    property_name =  FUEL:power_fraction
    component_name =  CH2
  [../]
  [./power_CH3]
    print_csv =  true
    data_type =  double
    property_name =  FUEL:power_fraction
    component_name =  CH3
  [../]
  [./MassFlowRateIn_SC_A]
    print_csv =  true
    data_type =  double
    property_name =  v_bc
    component_name =  MassFlowRateIn-SC-A
  [../]
  [./MassFlowRateIn_SC_B]
    print_csv =  true
    data_type =  double
    property_name =  v_bc
    component_name =  MassFlowRateIn-SC-B
  [../]
  [./Head_PumpB]
    print_csv =  true
    data_type =  double
    property_name =  Head
    component_name =  Pump-B
  [../]
  [./Head_PumpA]
    print_csv =  true
    data_type =  double
    property_name =  Head
    component_name =  Pump-A
  [../]
  [./friction1_SC_A]
    print_csv =  false
    data_type =  double
    property_name =  f
    component_name =  pipe1-SC-A
  [../]
  [./friction2_SC_A]
    print_csv =  false
    data_type =  double
    property_name =  f
    component_name =  pipe2-SC-A
  [../]
  [./friction1_SC_B]
    print_csv =  false
    data_type =  double
    property_name =  f
    component_name =  pipe1-SC-B
  [../]
  [./friction2_SC_B]
    print_csv =  false
    data_type =  double
    property_name =  f
    component_name =  pipe2-SC-B
  [../]
  [./friction1_CL_B]
    print_csv =  false
    data_type =  double
    property_name =  f
    component_name =  pipe1-CL-B
  [../]
  [./friction2_CL_B]
    print_csv =  false
    data_type =  double
    property_name =  f
    component_name =  pipe2-CL-B
  [../]
  [./friction1_CL_A]
    print_csv =  false
    data_type =  double
    property_name =  f
    component_name =  pipe1-CL-A
  [../]
  [./friction2_CL_A]
    print_csv =  false
    data_type =  double
    property_name =  f
    component_name =  pipe2-CL-A
  [../]
[]
[Monitored]
  #  [./sec_inlet_density]
  #    operator = ElementAverageValue
  #    path =
  #    data_type = double
  #    component_name = pipe1-SC-A
  #  [../]
  [./avg_temp_clad_CH1]
    operator =  ElementAverageValue
    path =  CLAD:TEMPERATURE
    data_type =  double
    component_name =  CH1
  [../]
  [./avg_temp_clad_CH2]
    operator =  ElementAverageValue
    path =  CLAD:TEMPERATURE
    data_type =  double
    component_name =  CH2
  [../]
  [./avg_temp_clad_CH3]
    # tests pressure monitoring in a pipe (ElementAverageValue operator)
    operator =  ElementAverageValue
    path =  CLAD:TEMPERATURE
    data_type =  double
    component_name =  CH3
  [../]
  [./avg_Fluid_Vel_H_L-A]
    # tests velocity monitoring in a pipe (ElementAverageValue operator)
    operator =  ElementAverageValue
    path =  VELOCITY
    data_type =  double
    component_name =  pipe1-HL-A
  [../]
  [./avg_Fluid_Vel_C_L_A]
    operator =  ElementAverageValue
    path =  VELOCITY
    data_type =  double
    component_name =  DownComer-A
  [../]
  [./avg_out_temp_sec_A]
    operator =  ElementAverageValue
    path =  TEMPERATURE
    data_type =  double
    component_name =  pipe2-SC-A
  [../]
  [./DownStreamSpeed]
    operator =  ElementAverageValue
    path =  VELOCITY
    data_type =  double
    component_name =  pipe1-CL-B
  [../]
  [./UpstreamSpeed]
    operator =  ElementAverageValue
    path =  VELOCITY
    data_type =  double
    component_name =  pipe1-CL-B
  [../]
  [./avg_temp_fuel_CH1]
    operator =  ElementAverageValue
    path =  FUEL:TEMPERATURE
    data_type =  double
    component_name =  CH1
  [../]
  [./avg_temp_fuel_CH2]
    operator =  ElementAverageValue
    path =  FUEL:TEMPERATURE
    data_type =  double
    component_name =  CH2
  [../]
  [./avg_temp_fuel_CH3]
    operator =  ElementAverageValue
    path =  FUEL:TEMPERATURE
    data_type =  double
    component_name =  CH3
  [../]
  [./sec_inlet_velocity]
    operator =  ElementAverageValue
    path =  VELOCITY
    data_type =  double
    component_name =  pipe1-SC-A
  [../]
[]
[Auxiliary]
  [./DG1_time_ratio]
    print_csv =  true
    data_type =  double
    initial_value =  0.0
  [../]
[./crew1DG2CoupledDG1]
  print_csv =  true
  data_type =  double
  initial_value =  0.0
[../]

  [./init_exp_frict]
    print_csv =  false
    data_type =  bool
    initial_value =  True
  [../]
  [./crew1DG1]
    print_csv =  true
    data_type =  bool
    initial_value =  False
  [../]
 [./crew1DG2CoupledDG1]
   print_csv =  true
   data_type =  bool
   initial_value =  False
 [../]
  [./crewSecPG]
    print_csv =  true
    data_type =  bool
    initial_value =  False
  [../]
 [./PrimPGrecovery]
   print_csv =  true
   data_type =  bool
   initial_value =  False
 [../]
  [./frict_m]
    print_csv =  false
    data_type =  double
    initial_value =  -1005.56
  [../]
  [./frict_q]
    print_csv =  false
    data_type =  double
    initial_value =  10005.1
  [../]
  [./scram_start_time]
    print_csv =  true
    data_type =  double
    initial_value =  101.0
  [../]
  [./friction_time_start_exp]
    print_csv =  false
    data_type =  double
    initial_value =  0.0
  [../]
  [./InitialMassFlowPrimary]
    print_csv =  true
    data_type =  double
    initial_value =  0
  [../]
  [./initialInletSecPress]
    print_csv =  false
    data_type =  double
    initial_value =  15219000
  [../]
  [./CladDamaged]
    print_csv =  true
    data_type =  bool
    initial_value =  False
  [../]
  [./DeltaTimeScramToAux]
    print_csv =  true
    data_type =  double
    initial_value =  200.0
  [../]
  [./InitialOutletSecPress]
    print_csv =  false
    data_type =  double
    initial_value =  151.7e5  #15170000
  [../]
  [./CladTempBranched]
    print_csv =  true
    data_type =  double
    initial_value = 1500.0
  [../]
  [./ScramStatus]
    print_csv =  true
    data_type =  bool
    initial_value =  false
  [../]
  [./AuxSystemUp]
    print_csv =  true
    data_type =  bool
    initial_value =  false
  [../]
  [./init_Power_Fraction_CH1]
    print_csv =  true
    data_type =  double
    initial_value =  3.33672612e-1
  [../]
  [./init_Power_Fraction_CH2]
    print_csv =  true
    data_type =  double
    initial_value =  3.69921461e-1
  [../]
  [./init_Power_Fraction_CH3]
    print_csv =  true
    data_type =  double
    initial_value =  2.96405926e-1
  [../]
 [./a_power_CH1]
 print_csv =  true
 data_type =  double
 initial_value =  3.33672612e-1
 [../]
 [./a_power_CH2]
 print_csv =  true
 data_type =  double
 initial_value =  3.69921461e-1
 [../]
 [./a_power_CH3]
 print_csv =  true
 data_type =  double
 initial_value =  2.96405926e-1
 [../]
 [./a_MassFlowRateIn_SC_A]
 print_csv =  true
 data_type =  double
 initial_value =  2.542
 [../]
 [./a_MassFlowRateIn_SC_B]
 print_csv =  true
 data_type =  double
 initial_value =  2.542
 [../]
 [./a_Head_PumpB]
 print_csv =  true
 data_type =  double
 initial_value =  8.9
 [../]
 [./a_Head_PumpA]
 print_csv =  true
 data_type =  double
 initial_value =  8.9
 [../]
 [./a_friction1_SC_A]
 print_csv =  false
 data_type =  double
 initial_value =  0.001
 [../]
 [./a_friction2_SC_A]
 print_csv =  false
 data_type =  double
 initial_value =  0.001
 [../]
 [./a_friction1_SC_B]
 print_csv =  false
 data_type =  double
 initial_value =  0.001
 [../]
 [./a_friction2_SC_B]
 print_csv =  false
 data_type =  double
 initial_value =  0.001
 [../]
 [./a_friction1_CL_B]
 print_csv =  false
 data_type =  double
 initial_value =  0.001
 [../]
 [./a_friction2_CL_B]
 print_csv =  true
 data_type =  double
 initial_value =  0.001
 [../]
 [./a_friction1_CL_A]
 print_csv =  false
 data_type =  double
 initial_value =  0.001
 [../]
 [./a_friction2_CL_A]
 print_csv =  true
 data_type =  double
 initial_value =  0.001
 [../]
 [./auxAbsolute]
   print_csv = true
   data_type = double
   initial_value = 0.001
 [../]
 [./DG1recoveryTime]
 data_type = double
 print_csv = true
 initial_value = 0.0
 [../]
 [./DG2recoveryTime]
 data_type = double
 print_csv = true
 initial_value = 0.0
 [../]
 [./SecPGrecoveryTime]
 data_type = double
 print_csv = true
 initial_value = 0.0
 [../]
 [./PrimPGrecoveryTime]
 data_type = double
 print_csv = true
 initial_value = 0.0
 [../]
 [./CladFailureDistThreshold]
 data_type = double
 print_csv = true
 initial_value = 0.0
 [../]
 [./crew1DG1Threshold]
 data_type = double
 print_csv = true
 initial_value = 0.0
 [../]
 [./crew1DG2CoupledDG1Threshold]
 data_type = double
 print_csv = true
 initial_value = 0.0
 [../]
 [./crewSecPGThreshold]
 data_type = double
 print_csv = true
 initial_value = 0.0
 [../]
 [./PrimPGrecoveryThreshold]
 data_type = double
 print_csv = true
 initial_value = 0.0
 [../]
 []

[TimeController]
 [./cntrAux]
 comparisonID = auxAbsolute
 time_step_size = 0.01
 referenceID = time
 delta = 0.5
 [../]
 []

 [RavenTools]
 [./PumpCoastDown]
 type = pumpCoastdownExponential
 #coefficient = 26.5
 coefficient = 1
 initial_flow_rate = 8.9
 [../]
 [./DecayHeatScalingFactor]
 type = decayHeat
 eq_type = 1
 initial_pow = 1
 operating_time = 20736000
 power_coefficient = 0.74
 [../]
 [./PumpCoastDownSec]
 type = pumpCoastdownExponential
 #coefficient = 10.5
 coefficient = 1
 initial_flow_rate = 1.0
 [../]
 []
