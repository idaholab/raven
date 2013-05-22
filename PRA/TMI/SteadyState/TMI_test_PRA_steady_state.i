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
  # close Functions section
  [./eos]
    type = NonIsothermalEquationOfState
    p_0 = 15.17e6 # Pa
    rho_0 = 738.350 # kg/m^3
    a2 = 1.e7 # m^2/s^2
    beta = .46e-3 # K^{-1}
    cv = 5.832e3 # J/kg-K
    e_0 = 3290122.80 # J/kg
    T_0 = 564.15 # K
  [../]
[]
 [Materials]
 [./fuel-mat]
 type = SolidMaterialProperties
 k = 3.65
 Cp = 288.734
 rho = 1.032e4
 [../]
 [./gap-mat]
 type = SolidMaterialProperties
 k = 1.084498
 Cp = 1.0
 rho = 1.
 [../]
 [./clad-mat]
 type = SolidMaterialProperties
 k = 16.48672
 Cp = 321.384
 rho = 6.55e3
 [../]
 [./wall-mat]
 type = SolidMaterialProperties
 k = 1.0
 rho = 80.0
 Cp = 4.0
 [../]
 []
[Components]
  [./reactor]
    type = Reactor
    initial_power = 2.77199979e9
  [../]

  # Core region components 
  [./CH1]
    # peak_power = '6.127004e8 0. 0.'
    type = CoreChannel
    eos = eos
    position = '0 -1.2 0'
    orientation = '0 0 1'
    A = 1.161864
    Dh = 0.01332254
    length = 3.6576
    n_elems = 8
    f = 0.01
    Hw = 5.33e4
    aw = 276.5737513
    Ts_init = 564.15
    n_heatstruct = 3
    name_of_hs = 'FUEL GAP CLAD'
    fuel_type = cylinder
    width_of_hs = '0.0046955  0.0000955  0.000673'
    elem_number_of_hs = '3 1 1'
    k_hs = '3.65 1.084498 16.48672'
    Cp_hs = '288.734 1.0 321.384'
    rho_hs = '1.0412e2 1.0 6.6e1'
    material_hs = 'fuel-mat gap-mat clad-mat'
    power_fraction = '3.33672612e-1 0 0'
  [../]
  [./CH2]
    # peak_power = '5.094461e8 0. 0.'
    type = CoreChannel
    eos = eos
    position = '0 0 0'
    orientation = '0 0 1'
    A = 1.549152542
    Dh = 0.01332254
    length = 3.6576
    n_elems = 8
    f = 0.01
    Hw = 5.33e4
    aw = 276.5737513
    Ts_init = 564.15
    n_heatstruct = 3
    name_of_hs = 'FUEL GAP CLAD'
    fuel_type = cylinder
    width_of_hs = '0.0046955  0.0000955  0.000673'
    elem_number_of_hs = '3 1 1'
    k_hs = '3.65  1.084498  16.48672'
    Cp_hs = '288.734  1.0  321.384'
 material_hs = 'fuel-mat gap-mat clad-mat'
    rho_hs = '1.0412e2 1. 6.6e1'
    power_fraction = '3.69921461e-1 0 0'
  [../]
  [./CH3]
    # peak_power = '3.401687e8 0. 0.'
    type = CoreChannel
    eos = eos
    position = '0 1.2 0'
    orientation = '0 0 1'
    A = 1.858983051
    Dh = 0.01332254
    length = 3.6576
    n_elems = 8
    f = 0.01
    Hw = 5.33e4
    aw = 276.5737513
    Ts_init = 564.15
    n_heatstruct = 3
    name_of_hs = 'FUEL GAP CLAD'
    fuel_type = cylinder
    width_of_hs = '0.0046955  0.0000955  0.000673'
    elem_number_of_hs = '3 1 1'
    k_hs = '3.65  1.084498  16.48672'
    Cp_hs = '288.734  1.0  6.6e3'
    rho_hs = '1.0412e2  1.0  6.6e1'
 material_hs = 'fuel-mat gap-mat clad-mat'
    power_fraction = '2.96405926e-1 0 0'
  [../]
  [./bypass_pipe]
    type = Pipe
    eos = eos
    position = '0 1.5 0'
    orientation = '0 0 1'
    A = 1.589571014
    Dh = 1.42264
    length = 3.6576
    n_elems = 5
    f = 0.001
    Hw = 0.0
  [../]
  [./LowerPlenum]
    type = ErgBranch
    eos = eos
    inputs = 'DownComer-A(out) DownComer-B(out)'
    outputs = 'CH1(in) CH2(in) CH3(in) bypass_pipe(in)'
    K = '0.2 0.2 0.2 0.2 0.4 40.0'
    Area = 3.618573408
    Initial_pressure = 151.7e5
  [../]
  [./UpperPlenum]
    type = ErgBranch
    eos = eos
    inputs = 'CH1(out) CH2(out) CH3(out) bypass_pipe(out)'
    outputs = 'pipe1-HL-A(in) pipe1-HL-B(in)'
    K = '0.5 0.5 0.5 80.0 0.5 0.5'
    Area = 7.562307456
    Initial_pressure = 151.7e5
  [../]
  [./DownComer-A]
    type = Pipe
    eos = eos
    position = '0 2.0 4.0'
    orientation = '0 0 -1'
    A = 3.6185734
    Dh = 1.74724302
    length = 4
    n_elems = 3
    f = 0.001
    Hw = 0.
  [../]
  [./pipe1-HL-A]
    type = Pipe
    eos = eos
    position = '0 0.5 4.0'
    orientation = '0 0 1'
    A = 7.562307456
    Dh = 3.103003207
    length = 4.
    n_elems = 3
    f = 0.001
    Hw = 0.0
  [../]
  [./pipe2-HL-A]
    type = Pipe
    eos = eos
    position = '0 0.5 8.0'
    orientation = '0 1 0'
    A = 2.624474
    Dh = 1.828
    length = 3.5
    n_elems = 3
    f = 0.001
    Hw = 0.0
  [../]
  [./pipe1-CL-A]
    type = Pipe
    eos = eos
    position = '0 3.0 4.0'
    orientation = '0 -1 0'
    A = 2.624474
    Dh = 1.828
    length = 1.
    n_elems = 3
    f = 0.001
    Hw = 0.0
  [../]
  [./pipe2-CL-A]
    type = Pipe
    eos = eos
    position = '0 4 4.0'
    orientation = '0 -1 0'
    A = 2.624474
    Dh = 1.828
    length = 0.8
    n_elems = 3
    f = 0.001
    Hw = 0.0
  [../]
  [./pipe1-SC-A]
    type = Pipe
    eos = eos
    position = '0 5.2 4.0'
    orientation = '0 -1 0'
    A = 2.624474
    Dh = 1.828
    length = 1.
    n_elems = 3
    f = 0.001
    Hw = 0.0
  [../]
  [./pipe2-SC-A]
    type = Pipe
    eos = eos
    position = '0 4.2 8.0'
    orientation = '0 1 0'
    A = 2.624474
    Dh = 1.828
    length = 1.
    n_elems = 3
    f = 0.001
    Hw = 0.0
  [../]
  [./Branch1-A]
    type = ErgBranch
    eos = eos
    inputs = 'pipe1-HL-A(out)'
    outputs = 'pipe2-HL-A(in) pipe-to-Pressurizer(in)'
    K = '0.5 0.7 80.'
    Area = 7.562307456
    Initial_pressure = 151.7e5
  [../]
  [./Branch2-A]
    type = ErgBranch
    eos = eos
    inputs = 'pipe1-CL-A(out)'
    outputs = 'DownComer-A(in)'
    K = '0.5 0.7'
    Area = 3.6185734
    Initial_pressure = 151.7e5
  [../]
  [./Branch3-A]
    type = ErgBranch
    eos = eos
    inputs = 'pipe2-HL-A(out)'
    outputs = 'HX-A(primary_in)'
    K = '0.5 0.7'
    Area = 2.624474
    Initial_pressure = 151.7e5
  [../]
  [./Pump-A]
    type = Pump
    eos = eos
    Area = 2.624474
    Initial_pressure = 151.7e5
    Head = 8.9
    K_reverse = 1000
    outputs = 'pipe1-CL-A(in)'
    inputs = 'pipe2-CL-A(out)'
  [../]
  [./HX-A]
    type = HeatExchanger
    eos = eos
    eos_secondary = eos
    position = '0 4. 8.'
    orientation = '0 0 -1'
    A = 5.
    A_secondary = 5.
    Dh = 0.01
    Dh_secondary = 0.01
    length = 4.
    n_elems = 10
    Hw = 1.e4
    Hw_secondary = 1.e4
    aw = 539.02
    aw_secondary = 539.02
    f = 0.01
    f_secondary = 0.01
    Twall_init = 564.15
    wall_thickness = 0.001
    k_wall = 100.0
    rho_wall = 100.0
    Cp_wall = 100.0
    n_wall_elems = 2
 material_wall = wall-mat
  [../]
  [./Branch4-A]
    type = ErgBranch
    eos = eos
    inputs = 'pipe1-SC-A(out)'
    outputs = 'HX-A(secondary_in)'
    K = '0.5 0.7'
    Area = 2.624474e2
    Initial_pressure = 151.7e5
  [../]
  [./Branch5-A]
    type = ErgBranch
    eos = eos
    inputs = 'HX-A(secondary_out)'
    outputs = 'pipe2-SC-A(in)'
    K = '0.5 0.7'
    Area = 2.624474e2
    Initial_pressure = 151.7e5
  [../]
  [./Branch6-A]
    type = ErgBranch
    eos = eos
    inputs = 'HX-A(primary_out)'
    outputs = 'pipe2-CL-A(in)'
    K = '0.5 0.7'
    Area = 2.624474e2
    Initial_pressure = 151.7e5
  [../]
  [./PressureOutlet-SC-A]
    type = TimeDependentVolume
    input = 'pipe2-SC-A(out)'
    p_bc = '151.7e5'
    T_bc = 564.15
    eos = eos
  [../]
  [./DownComer-B]
    type = Pipe
    eos = eos
    position = '0 -2.0 4.0'
    orientation = '0 0 -1'
    A = 3.6185734
    Dh = 1.74724302
    length = 4
    n_elems = 3
    f = 0.001
    Hw = 0.
  [../]
  [./pipe1-HL-B]
    type = Pipe
    eos = eos
    position = '0 -0.5 4.0'
    orientation = '0 0 1'
    A = 7.562307456
    Dh = 3.103003207
    length = 4.
    n_elems = 3
    f = 0.001
    Hw = 0.0
  [../]
  [./pipe2-HL-B]
    type = Pipe
    eos = eos
    position = '0 -0.5 8.0'
    orientation = '0 -1 0'
    A = 2.624474
    Dh = 1.828
    length = 3.5
    n_elems = 3
    f = 0.001
    Hw = 0.0
  [../]
  [./pipe1-CL-B]
    type = Pipe
    eos = eos
    position = '0 -3.0 4.0'
    orientation = '0 1 0'
    A = 2.624474
    Dh = 1.828
    length = 1.
    n_elems = 3
    f = 0.001
    Hw = 0.0
  [../]
  [./pipe2-CL-B]
    type = Pipe
    eos = eos
    position = '0 -4.0 4.0'
    orientation = '0 1 0'
    A = 2.624474
    Dh = 1.828
    length = 0.8
    n_elems = 3
    f = 0.001
    Hw = 0.0
  [../]
  [./pipe1-SC-B]
    type = Pipe
    eos = eos
    position = '0 -5.2 4.0'
    orientation = '0 1 0'
    A = 2.624474
    Dh = 1.828
    length = 1.
    n_elems = 3
    f = 0.001
    Hw = 0.0
  [../]
  [./pipe2-SC-B]
    type = Pipe
    eos = eos
    position = '0 -4.2 8.0'
    orientation = '0 -1 0'
    A = 2.624474
    Dh = 1.828
    length = 1.
    n_elems = 3
    f = 0.001
    Hw = 0.0
  [../]
  [./Branch1-B]
    type = ErgBranch
    eos = eos
    inputs = 'pipe1-HL-B(out)'
    outputs = 'pipe2-HL-B(in)'
    K = '0.5 0.7'
    Area = 7.562307456
    Initial_pressure = 151.7e5
  [../]
  [./Branch2-B]
    type = ErgBranch
    eos = eos
    inputs = 'pipe1-CL-B(out)'
    outputs = 'DownComer-B(in)'
    K = '0.5 0.7'
    Area = 3.6185734
    Initial_pressure = 151.7e5
  [../]
  [./Branch3-B]
    type = ErgBranch
    eos = eos
    inputs = 'pipe2-HL-B(out)'
    outputs = 'HX-B(primary_in)'
    K = '0.5 0.7'
    Area = 2.624474
    Initial_pressure = 151.7e5
  [../]
  [./Pump-B]
    type = Pump
    eos = eos
    Area = 2.624474
    Initial_pressure = 151.7e5
    Head = 8.9
    K_reverse = 1000
    outputs = 'pipe1-CL-B(in)'
    inputs = 'pipe2-CL-B(out)'
  [../]
  [./HX-B]
    type = HeatExchanger
    eos = eos
    eos_secondary = eos
    position = '0 -4. 8.'
    orientation = '0 0 -1'
    A = 5.
    A_secondary = 5.
    Dh = 0.01
    Dh_secondary = 0.01
    length = 4.
    n_elems = 10
    Hw = 1.e4
    Hw_secondary = 1.e4
    aw = 539.02
    aw_secondary = 539.02
    f = 0.01
    f_secondary = 0.01
    Twall_init = 564.15
    wall_thickness = 0.001
    k_wall = 100.0
    rho_wall = 100.0
    Cp_wall = 100.0
    material_wall = wall-mat
    n_wall_elems = 2
    disp_mode = -1.0
  [../]
  [./Branch4-B]
    type = ErgBranch
    eos = eos
    inputs = 'pipe1-SC-B(out)'
    outputs = 'HX-B(secondary_in)'
    K = '0.5 0.7'
    Area = 2.624474e2
    Initial_pressure = 151.7e5
  [../]
  [./Branch5-B]
    type = ErgBranch
    eos = eos
    inputs = 'HX-B(secondary_out)'
    outputs = 'pipe2-SC-B(in)'
    K = '0.5 0.7'
    Area = 2.624474e2
    Initial_pressure = 151.7e5
  [../]
  [./Branch6-B]
    type = ErgBranch
    eos = eos
    inputs = 'HX-B(primary_out)'
    outputs = 'pipe2-CL-B(in)'
    K = '0.5 0.7'
    Area = 2.624474e2
    Initial_pressure = 151.7e5
  [../]
  [./PressureOutlet-SC-B]
    type = TimeDependentVolume
    input = 'pipe2-SC-B(out)'
    p_bc = '151.7e5'
    T_bc = 564.15
    eos = eos
  [../]
  [./pipe-to-Pressurizer]
    type = Pipe
    eos = eos
    position = '0 0.5 8.0'
    orientation = '0 0 1'
    A = 2.624474
    Dh = 1.828
    length = 0.5
    n_elems = 3
    f = 10.
    Hw = 0.0
  [../]
  [./Pressurizer]
    type = TimeDependentVolume
    input = 'pipe-to-Pressurizer(out)'
    p_bc = '151.7e5'
    T_bc = 564.15
    eos = eos
  [../]
 # [./high_pressure_seconday_A]
 #   T_bc = 537.15
 #   p_bc = '152.19e5'
 #   eos = eos
 #   input = 'pipe1-SC-A(in)'
 #   type = TimeDependentVolume
 # [../]
 #  [./high_pressure_seconday_B]
 #   T_bc = 537.15
 #   p_bc = '152.19e5'
 #   eos = eos
 #   input = 'pipe1-SC-B(in)'
 #   type = TimeDependentVolume
 # [../]
 [./MassFlowRateIn-SC-B]
 #type = TDM
 type = TimeDependentJunction
 input = 'pipe1-SC-B(in)'
 #	massflowrate_bc = 8801.1
 v_bc = 2.542 #4.542
 T_bc = 537.15
 eos = eos
 [../] 
 [./MassFlowRateIn-SC-A]
 #type = TDM
 type = TimeDependentJunction
 input = 'pipe1-SC-A(in)'
 #	massflowrate_bc = 8801.1
 v_bc = 2.542 #4.542
 T_bc = 537.15
 eos = eos
 [../]
[]

[Preconditioning]
# active = 'FDP_Newton'
# End preconditioning block
active = 'SMP'
[./SMP]
type = SMP
full = true
petsc_options = '-snes_mf_operator'
[../]
[./FDP_PJFNK]
# These options **together** cause a zero pivot in this problem, even without SUPG terms.
# But using either option alone appears to be OK.
# petsc_options_iname = '-mat_fd_coloring_err -mat_fd_type'
# petsc_options_value = '1.e-10               ds'
type = FDP
full = true
petsc_options = '-snes_mf_operator -pc_factor_shift_nonzero'
petsc_options_iname = '-mat_fd_type'
petsc_options_value = 'ds'
petsc_options_iname = '-mat_fd_type'
petsc_options_value = 'ds'
[../]
[./FDP_Newton]
# These options **together** cause a zero pivot in this problem, even without SUPG terms.
# But using either option alone appears to be OK.
# petsc_options_iname = '-mat_fd_coloring_err -mat_fd_type'
# petsc_options_value = '1.e-10               ds'
type = FDP
full = true
petsc_options = '-snes'
petsc_options_iname = '-mat_fd_type'
petsc_options_value = 'ds'
petsc_options_iname = '-mat_fd_type'
petsc_options_value = 'ds'
[../]
[]
[Executioner]
# restart_file_base = TMI_test_PRA_out_restart_0194
type = RavenExecutioner
dt = 5e-2
[./TimeStepper]
  type = FunctionDT
  time_t = '0         1.0    3.0       5.01       9.5     9.75    14        17     5e1'
  time_dt = '1.e-3  0.025    0.026   2.5e-2    2.5e-2    2.5e-2   3.5e-2    5.5e-2 5.5e-2'
[../]
dtmax = 9999
e_tol = 10.0
e_max = 99999.
max_increase = 3
perf_log = true
petsc_options_iname = '-ksp_gmres_restart -pc_type'
petsc_options_value = '300 lu' # '300'
nl_rel_tol = 1e-6
nl_abs_tol = 1e-10
nl_max_its = 100
l_tol = 1e-5 # Relative linear tolerance for each Krylov solve
l_max_its = 100 # Number of linear iterations for each Krylov solve
start_time = 0.0
end_time = 60.0
ss_check_tol = 1e-05
nl_rel_step_tol = 1e-3
predictor_scale = 0.6
[./Quadrature]
type = TRAP
order = FIRST
[../]
[]


[Output]
  # xda = true
  file_base = TMI_test_PRA_steady_state
  exodus = true
  output_initial = true
  perf_log = true
  num_restart_files = 1
[]

 [Controlled]
 # control logic file name
 #control_logic_input = TMI_PRA_trans_MC_control
 [./power_CH1]
 print_csv = true
 property_name = FUEL:power_fraction
 data_type = double
 component_name = CH1
 [../]
 [./power_CH2]
 property_name = FUEL:power_fraction
 print_csv = true
 data_type = double
 component_name = CH2
 [../]
 [./power_CH3]
 property_name = FUEL:power_fraction
 print_csv = true
 data_type = double
 component_name = CH3
 [../]
 #  [./high_pressure_secondary_A]
 #   property_name = p_bc
 #   print_csv = true
 #   data_type = double
 #   component_name = high_pressure_seconday_A
 # [../]
 # [./high_pressure_secondary_B]
 #   property_name = p_bc
 #   print_csv = true
 #   data_type = double
 #   component_name = high_pressure_seconday_B
 # [../]
 [./MassFlowRateIn_SC_A]
 property_name = v_bc
 print_csv = true
 data_type = double
 component_name = MassFlowRateIn-SC-A
 [../]
 [./MassFlowRateIn_SC_B]
 property_name = v_bc
 print_csv = true
 data_type = double
 component_name = MassFlowRateIn-SC-B
 [../] 
 [./Head_PumpB]
 property_name = Head
 data_type = double
 print_csv = true
 component_name = Pump-B
 [../]
 [./Head_PumpA]
 property_name = Head
 data_type = double
 print_csv = true
 component_name = Pump-A
 [../]
 [./friction1_SC_A]
 print_csv = false
 property_name = f
 data_type = double
 component_name = pipe1-SC-A
 [../]
 [./friction2_SC_A]
 print_csv = false
 property_name = f
 data_type = double
 component_name = pipe2-SC-A
 [../]
 [./friction1_SC_B]
 print_csv = false
 property_name = f
 data_type = double
 component_name = pipe1-SC-B
 [../]
 [./friction2_SC_B]
 print_csv = false
 property_name = f
 data_type = double
 component_name = pipe2-SC-B
 [../]
 [./friction1_CL_B]
 print_csv = false
 property_name = f
 data_type = double
 component_name = pipe1-CL-B
 [../]
 [./friction2_CL_B]
 print_csv = false
 property_name = f
 data_type = double
 component_name = pipe2-CL-B
 [../]
 [./friction1_CL_A]
 print_csv = false
 property_name = f
 data_type = double
 component_name = pipe1-CL-A
 [../]
 [./friction2_CL_A]
 print_csv = false
 property_name = f
 data_type = double
 component_name = pipe2-CL-A
 [../]
 []
 
 [Monitored]
 [./avg_temp_clad_CH1]
 operator = ElementAverageValue
 path = CLAD:TEMPERATURE
 data_type = double
 component_name = CH1
 [../]
 [./avg_temp_clad_CH2]
 operator = ElementAverageValue    
 path = CLAD:TEMPERATURE
 data_type = double
 component_name = CH2
 [../]
 [./avg_temp_clad_CH3]
 # tests pressure monitoring in a pipe (ElementAverageValue operator)
 operator = ElementAverageValue
 path = CLAD:TEMPERATURE
 data_type = double
 component_name = CH3
 [../]
 [./avg_Fluid_Vel_H_L-A]
 # tests velocity monitoring in a pipe (ElementAverageValue operator)
 operator = ElementAverageValue
 path = VELOCITY
 data_type = double
 component_name = pipe1-HL-A
 [../]
 [./avg_Fluid_Vel_C_L_A]
 operator = ElementAverageValue
 path = VELOCITY
 data_type = double
 component_name = DownComer-A
 [../]
 [./avg_out_temp_sec_A]
 operator = ElementAverageValue
 path = TEMPERATURE
 data_type = double
 component_name = pipe2-SC-A
 [../]
 [./DownStreamSpeed]
 operator = ElementAverageValue
 path = VELOCITY
 data_type = double
 component_name = pipe1-CL-B
 [../]
 [./UpstreamSpeed]
 operator = ElementAverageValue
 path = VELOCITY
 data_type = double
 component_name = pipe1-CL-B
 [../]
 [./avg_temp_fuel_CH1]
 operator = ElementAverageValue
 path = FUEL:TEMPERATURE
 data_type = double
 component_name = CH1
 [../]
 [./avg_temp_fuel_CH2]
 operator = ElementAverageValue
 path = FUEL:TEMPERATURE
 data_type = double
 component_name = CH2
 [../]
 [./avg_temp_fuel_CH3]
 operator = ElementAverageValue
 path = FUEL:TEMPERATURE
 data_type = double
 component_name = CH3
 [../]
 [./sec_inlet_velocity]
 operator = ElementAverageValue
 path = VELOCITY
 data_type = double
 component_name = pipe1-SC-A
 [../]
 #  [./sec_inlet_density]
 #    operator = ElementAverageValue
 #    path = 
 #    data_type = double
 #    component_name = pipe1-SC-A
 #  [../]
 []
 
 [Distributions]
 RNG_seed = 20021986
 [./auxBackUpTimeDist]
 type = NormalDistribution
 mu = 20.0
 sigma = 6.0
 xMin  = 2.0
 xMax  = 40.0
 truncation = 1
 [../]
 [./noise]
 type = NormalDistribution
 mu = 0.0
 sigma = 0.05
 xMin  = -1
 xMax  = 1
 truncation = 1
 [../]
 [./CladFailureDist]
 type = TriangularDistribution
 xMin = 1255.3722 # Lower bound (PRA succes criteria)
 xPeak = 1477.59
 xMax = 1699.8167 # Upper bound (Urbanic-Heidrick Transition Temperature)
 truncation = 1
 lowerBound = 1255.3722
 upperBound = 1699.8167 
 [../]
 []
 
 [RavenAuxiliary]
 [./init_exp_frict]
 data_type = bool
 print_csv = false
 initial_value = True
 [../]
 [./frict_m]
 data_type = double
 initial_value = -1505.56
 print_csv = false
 [../]
 [./frict_q]
 data_type = double
 initial_value = 15005.1
 print_csv = false
 [../]
 [./scram_start_time]
 data_type = double
 initial_value = 101.0
 print_csv = true
 [../]
 [./friction_time_start_exp]
 data_type = double
 initial_value = 0.0
 print_csv = false
 [../]
 [./InitialMassFlowPrimary]
 data_type = double
 initial_value = 0
 print_csv = true
 [../]
 [./initialInletSecPress]
 data_type = double
 print_csv = false
 initial_value = 15219000
 [../]
 [./CladDamaged]
 data_type = bool
 print_csv = true
 initial_value = False
 [../]
 [./DeltaTimeScramToAux]
 data_type = double
 initial_value = 200.0
 print_csv = true
 [../]
 [./InitialOutletSecPress]
 data_type = double
 print_csv = false
 initial_value = 151.7e5  #15170000
 [../]
 [./CladTempTreshold]
 data_type = double
 print_csv = true
 initial_value = 1477.59
 [../]
 [./ScramStatus]
 print_csv = true
 data_type = bool
 initial_value = false
 [../]
 [./init_Power_Fraction_CH1]
 print_csv = true
 data_type = double
 initial_value = 3.33672612e-1 
 [../]
 [./init_Power_Fraction_CH2]
 print_csv = true
 data_type = double
 initial_value = 3.69921461e-1
 [../]
 [./init_Power_Fraction_CH3]
 print_csv = true
 data_type = double
 initial_value = 2.96405926e-1 
 [../]  
 
 []
