[GlobalParams]
# 2=2 eqn, 1D isothermal flow
# 3=3 eqn, 1D non-isothermal flow
# 7=7 eqn, 1D 2-phase flow
 model_type = 3
 
 global_init_P = 15.17e6
 global_init_V = 1.
 global_init_T = 564.15
 
#  scaling_factor_var = '1. 1.e-6 1.e-7'
 scaling_factor_var   = '1e-3 1e-4 1e-8'
 temperature_sf = '1e-4'
 stabilization_type = 'LAPIDUS'
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
 type = SolidMaterialProperties
 k = 3.65
 Cp = 288.734
 rho = 10.31e3 #1.0412e2
 [../]
 [./gap-mat]
 type = SolidMaterialProperties
 k = 1.084498
 Cp = 12 # 1.0
 rho = 2.22 # 1.0
 [../]
 [./clad-mat]
 type = SolidMaterialProperties
 k = 16.48672
 Cp = 6.6e3 # 321.384
 rho = 6.6e3 #6.6e1
 [../]
 [./clad3-mat]
 type = SolidMaterialProperties
 k = 16.48672
 Cp = 6.6e3
 rho = 6.6e3 #6.6e1
 [../]
 [./wall-mat]
 type = SolidMaterialProperties
 k = 15 # 100.0
 rho = 6.6e3 # 100.0
 Cp = 6.6e3 # 100.0
 [../]
 []
 
 
[Components]
 [./reactor]
 type = Reactor
 initial_power = 2.77199979e9
 [../]
 
#Core region components #############################################################
 [./CH1]
 type = CoreChannel
 eos = eos
 position = '0 -1.2 0'
 orientation = '0 0 1'
 A = 1.161864
 Dh = 0.01332254
 length = 3.6576
 n_elems = 16 #8
 
 f = 0.01
 Hw = 5.33e4
# aw = 276.5737513
 Phf = 321.341084980423
 Ts_init = 564.15
 
 dim_hs = 1
 n_heatstruct = 3
 name_of_hs = 'FUEL GAP CLAD'
 fuel_type = cylinder
 width_of_hs = '0.0046955  0.0000955  0.000673'
 elem_number_of_hs = '3 1 1'
 material_hs = 'fuel-mat gap-mat clad-mat'
#peak_power = '6.127004e8 0. 0.'
 power_fraction = '3.33672612e-1 0 0'
 [../]
 
 [./CH2]
 type = CoreChannel
 eos = eos
 position = '0 0 0'
 orientation = '0 0 1'
 A = 1.549152542
 Dh = 0.01332254
 length = 3.6576
 n_elems = 16 #8
 
 f = 0.01
 Hw = 5.33e4
# aw = 276.5737513
 Phf = 428.454929876871
 Ts_init = 564.15
 
 dim_hs = 1
 n_heatstruct = 3
 name_of_hs = 'FUEL GAP CLAD'
 fuel_type = cylinder
 width_of_hs = '0.0046955  0.0000955  0.000673'
 elem_number_of_hs = '3 1 1'
 material_hs = 'fuel-mat gap-mat clad-mat'
#peak_power = '5.094461e8 0. 0.'
 power_fraction = '3.69921461e-1 0 0'
 [../]
 
 [./CH3]
 type = CoreChannel
 eos = eos
 position = '0 1.2 0'
 orientation = '0 0 1'
 A = 1.858983051
 Dh = 0.01332254
 length = 3.6576
 n_elems = 16 #8
 
 f = 0.01
 Hw = 5.33e4
# aw = 276.5737513
 Phf = 514.145916018189
 Ts_init = 564.15
 
 dim_hs = 1
 n_heatstruct = 3
 name_of_hs = 'FUEL GAP CLAD'
 fuel_type = cylinder
 width_of_hs = '0.0046955  0.0000955  0.000673'
 elem_number_of_hs = '3 1 1'
 material_hs = 'fuel-mat gap-mat clad3-mat'
#peak_power = '3.401687e8 0. 0.'
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
 type = Branch
 eos = eos
 inputs = 'DownComer-A(out) DownComer-B(out)'
 outputs = 'CH1(in) CH2(in) CH3(in) bypass_pipe(in)'
 K = '0.2 0.2 0.2 0.2 0.4 40.0'
 Area = 3.618573408
 [../]
 
 [./UpperPlenum]
 type = Branch
 eos = eos
 inputs = 'CH1(out) CH2(out) CH3(out) bypass_pipe(out)'
 outputs = 'pipe1-HL-A(in) pipe1-HL-B(in)'
 K = '0.5 0.5 0.5 80.0 0.5 0.5'
 Area = 7.562307456
 [../]
##############################################################################################
 
#Loop A components ###########################################################################
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
 type = Branch
 eos = eos
 inputs = 'pipe1-HL-A(out)'
 outputs = 'pipe2-HL-A(in) pipe-to-Pressurizer(in)'
 K = '0.5 0.7 80.'
 Area = 7.562307456
 [../]
 
 [./Branch2-A]
 type = Branch
 eos = eos
 inputs = 'pipe1-CL-A(out)'
 outputs = 'DownComer-A(in)'
 K = '0.5 0.7'
 Area = 3.6185734
 
 [../]
 
 [./Branch3-A]
 type = Branch
 eos = eos
 inputs = 'pipe2-HL-A(out)'
 outputs = 'HX-A(primary_in)'
 K = '0.5 0.7'
 Area = 2.624474
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
 
# aw = 539.02
# aw_secondary = 539.02
 Phf = 2695.100000000000
 Phf_secondary = 2695.100000000000
 
 f = 0.01
 f_secondary = 0.01
 
 dim_wall = 1
 Twall_init = 564.15
 wall_thickness = 0.001
 n_wall_elems = 2
 material_wall = wall-mat
 [../]
 
 [./Branch4-A]
 type = Branch
 eos = eos
 inputs = 'pipe1-SC-A(out)'
 outputs = 'HX-A(secondary_in)'
 K = '0.5 0.7'
 Area = 2.624474e2
 [../]
 
 [./Branch5-A]
 type = Branch
 eos = eos
 inputs = 'HX-A(secondary_out)'
 outputs = 'pipe2-SC-A(in)'
 K = '0.5 0.7'
 Area = 2.624474e2
 [../]
 
 [./Branch6-A]
 type = Branch
 eos = eos
 inputs = 'HX-A(primary_out)'
 outputs = 'pipe2-CL-A(in)'
 K = '0.5 0.7'
 Area = 2.624474e2
 [../]
 
 [./MassFlowRateIn-SC-A]
#type = TDM
 type = TimeDependentJunction
 input = 'pipe1-SC-A(in)'
#   massflowrate_bc = 8801.1
 v_bc = 4.542
 T_bc = 537.15
 eos = eos
 [../]
 [./PressureOutlet-SC-A]
 type = TimeDependentVolume
 input = 'pipe2-SC-A(out)'
 p_bc = '151.7e5'
 T_bc = 564.15
 eos = eos
 [../]
##############################################################################################
 
#Loop B components ###########################################################################
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
 type = Branch
 eos = eos
 inputs = 'pipe1-HL-B(out)'
 outputs = 'pipe2-HL-B(in)'
 K = '0.5 0.7'
 Area = 7.562307456
 [../]
 
 [./Branch2-B]
 type = Branch
 eos = eos
 inputs = 'pipe1-CL-B(out)'
 outputs = 'DownComer-B(in)'
 K = '0.5 0.7'
 Area = 3.6185734
 [../]
 
 [./Branch3-B]
 type = Branch
 eos = eos
 inputs = 'pipe2-HL-B(out)'
 outputs = 'HX-B(primary_in)'
 K = '0.5 0.7'
 Area = 2.624474
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
 
# aw = 539.02
# aw_secondary = 539.02
 Phf = 2695.100000000000
 Phf_secondary = 2695.100000000000
 
 f = 0.01
 f_secondary = 0.01
 
 dim_wall = 1
 Twall_init = 564.15
 wall_thickness = 0.001
 material_wall = wall-mat
 n_wall_elems = 2
 
 disp_mode = -1.0
 [../]
 
 [./Branch4-B]
 type = Branch
 eos = eos
 inputs = 'pipe1-SC-B(out)'
 outputs = 'HX-B(secondary_in)'
 K = '0.5 0.7'
 Area = 2.624474e2
 [../]
 
 [./Branch5-B]
 type = Branch
 eos = eos
 inputs = 'HX-B(secondary_out)'
 outputs = 'pipe2-SC-B(in)'
 K = '0.5 0.7'
 Area = 2.624474e2
 [../]
 
 [./Branch6-B]
 type = Branch
 eos = eos
 inputs = 'HX-B(primary_out)'
 outputs = 'pipe2-CL-B(in)'
 K = '0.5 0.7'
 Area = 2.624474e2
 [../]
 
 [./MassFlowRateIn-SC-B]
#type = TDM
 type = TimeDependentJunction
 input = 'pipe1-SC-B(in)'
#   massflowrate_bc = 8801.1
 v_bc = 4.542
 T_bc = 537.15
 eos = eos
 [../]
 [./PressureOutlet-SC-B]
 type = TimeDependentVolume
 input = 'pipe2-SC-B(out)'
 p_bc = '151.7e5'
 T_bc = 564.15
 eos = eos
 [../]
##############################################################################################
 
 
# Pressurizer ################################################################################
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
##############################################################################################
 []

[Preconditioning]
# active = 'SMP_Newton'
 active = 'SMP_PJFNK'
# active = 'FDP_PJFNK'
# active = 'FDP_Newton'
 
# The definitions of the above-named blocks follow.
 [./SMP_PJFNK]
 type = SMP
 full = true
 
# Preconditioned JFNK (default)
 solve_type = 'PJFNK'
 petsc_options_iname = '-mat_fd_type  -mat_mffd_type'
 petsc_options_value = 'ds             ds'
 [../]
 
 [./SMP_Newton]
 type = SMP
 full = true
 solve_type = 'NEWTON'
 [../]
 
 [./FDP_PJFNK]
 type = FDP
 full = true
 
# Preconditioned JFNK (default)
 solve_type = 'PJFNK'
 petsc_options_iname = '-mat_fd_coloring_err'
 petsc_options_value = '1.e-10'
 petsc_options_iname = '-mat_fd_type'
 petsc_options_value = 'ds'
 [../]
 
 [./FDP_Newton]
 type = FDP
 full = true
 solve_type = 'NEWTON'
 
 petsc_options_iname = '-mat_fd_coloring_err'
 petsc_options_value = '1.e-10'
 petsc_options_iname = '-mat_fd_type'
 petsc_options_value = 'ds'
 [../]
 [] # End preconditioning block

[Executioner]
  # petsc_options_iname = '-ksp_gmres_restart -pc_type'
  # '300'
  # num_steps = '3'
  # time_t =  ' 0      1.0        3.0         5.01       9.5       9.75    14          17        60       61.1     100.8    101.5  102.2 120.0  400 1000 1.0e5'
  # time_dt =  '1.e-1  0.1        0.15        0.20       0.25      0.30    0.35        0.40    0.45      0.09      0.005     0.008   0.2   0.2    0.2 0.3  0.6'
  nl_abs_tol = 1e-4 #5e-5
  restart_file_base = 0957
   nl_rel_tol = 1e-8 #1e-9
  ss_check_tol = 1e-05
  nl_max_its = 120
  type = RavenExecutioner
  petsc_options_value = lu # '300'
  l_max_its = 100 # Number of linear iterations for each Krylov solve
  start_time = 100.0
  [./Predictor]
    type = SimplePredictor
    scale = 0.6
  [../]
  dtmax = 9999
  nl_rel_step_tol = 1e-3
  dt = 5e-5
  petsc_options_iname = -pc_type
  l_tol = 1e-4 # Relative linear tolerance for each Krylov solve
  end_time = 2500.0
# [./TimeStepper]
# type = SolutionTimeAdaptiveDT
# dt = 1.0
# [../]
 
[./TimeStepper]
    type = FunctionDT
    time_t = ' 0      1.0        3.0         5.01       9.5       9.75    14          17        120.0  2501.23 1.0e5'
    time_dt = '1.e-1  0.1        0.15        0.20       0.25      0.30    1        1.2    2      5  50'
  [../]
  [./Quadrature]
    type = TRAP
    order = FIRST
  [../]
[]

[Outputs]
  # xda = true
  # num_restart_files = 1
  output_initial = true
  exodus = false 
  file_base = TMI_test_PRA_transient_less_w_out
  csv = true
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
 print_csv =  false
 data_type =  double
 property_name =  v_bc
 component_name =  MassFlowRateIn-SC-B
 [../]
 [./Head_PumpB]
 print_csv =  false
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
 print_csv =  true
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
#  [./avg_out_temp_sec_A]
#    operator =  ElementAverageValue
#    path =  TEMPERATURE
#    data_type =  double
#    component_name =  pipe2-SC-A
#  [../]
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
#  [./avg_temp_fuel_CH2]
#    operator =  ElementAverageValue
#    path =  FUEL:TEMPERATURE
#    data_type =  double
#    component_name =  CH2
#  [../]
#  [./avg_temp_fuel_CH3]
#    operator =  ElementAverageValue
#    path =  FUEL:TEMPERATURE
#    data_type =  double
#    component_name =  CH3
#  [../]
 [./sec_inlet_velocity]
 operator =  ElementAverageValue
 path =  VELOCITY
 data_type =  double
 component_name =  pipe1-SC-A
 [../]
 []
[Distributions]
 RNG_seed = 1
 [./crewSecPG]
 type = NormalDistribution
 mu = 1400
 sigma = 400
 xMin = 0
 xMax = 2500
 [../]
 [./PrimPGrecovery]
 type = NormalDistribution
 mu = 2000
 sigma = 500
 xMin = 0
 xMax = 2500
 [../]
 [./crew1DG1]
 type = NormalDistribution
 mu = 800
 sigma = 200
 xMin = 0
 xMax = 2500
 [../]
 [./CladFailureDist]
 type = TriangularDistribution
 xMin = 1055.3722
 xPeak = 1277.59
 xMax = 1499.8167
 truncation = 1
 lowerBound = 1055.3722
 upperBound = 1499.8167
 [../]
 []
[Auxiliary]
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
 initial_value =  -500.56
 [../]
 [./frict_q]
 print_csv =  false
 data_type =  double
 initial_value =  5000
 [../]
#[./frict_m]
#print_csv =  false
#data_type =  double
#initial_value =  -1704.56
#[../]
#[./frict_q]
#print_csv =  false
#data_type =  double
#initial_value =  15105.1
#[../]
 [./scram_start_time]
 print_csv =  false
 data_type =  double
 initial_value =  151.0
 [../]
 [./friction_time_start_exp]
 print_csv =  false
 data_type =  double
 initial_value =  0.0
 [../]
 [./CladDamaged]
 print_csv =  true
 data_type =  bool
 initial_value =  False
 [../]
 [./CladTempBranched]
 print_csv =  true
 data_type =  double
 initial_value = 0.0
 [../]
 [./ScramStatus]
 print_csv =  false
 data_type =  bool
 initial_value =  false
 [../]
 [./AuxSystemUp]
 print_csv =  true
 data_type =  bool
 initial_value =  false
 [../]
 [./init_Power_Fraction_CH1]
 print_csv =  false
 data_type =  double
 initial_value =  3.33672612e-1
 [../]
 [./init_Power_Fraction_CH2]
 print_csv =  false
 data_type =  double
 initial_value =  3.69921461e-1
 [../]
 [./init_Power_Fraction_CH3]
 print_csv =  false
 data_type =  double
 initial_value =  2.96405926e-1
 [../]
 [./DG1recoveryTime]
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
 [./deltaAux]
 data_type = double
 print_csv = true
 initial_value = 0.0
 [../]
 [./crew1DG1Threshold]
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
 [./keepGoing]
 data_type = bool
 print_csv = true
 initial_value = true
 [../]
 []
#[TimeController]
# [./cntrAux]
# comparisonID = auxAbsolute
# time_step_size = 0.01
# referenceID = time
# delta = 0.5
# [../]
# []
 
 [RavenTools]
 [./PumpCoastDown]
 type = pumpCoastdownExponential
 coefficient = 26.5
 initial_flow_rate = 8.9
 [../]
 [./DecayHeatScalingFactor]
 type = decayHeat
 eq_type = 1
 initial_pow = 1
 operating_time = 20736000
 power_coefficient = 0.54
 start_time = 0.0
 [../]
 [./PumpCoastDownSec]
 type = pumpCoastdownExponential
 coefficient = 10.5
 initial_flow_rate = 1.0
 [../]
 []
