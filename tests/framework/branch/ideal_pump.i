[GlobalParams]
  # 2=2 eqn, 1D isothermal flow
  # 3=3 eqn, 1D non-isothermal flow
  # 7=7 eqn, 1D 2-phase flow
  global_init_P = 1.e5
  global_init_V = 0.0
  global_init_T = 300.
  model_type = 3
  stabilization_type = NONE
[]
[EoS]
  # close Functions section
  [./eos]
    type = NonIsothermalEquationOfState
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
    eos = eos
    position = '0 0 0'
    orientation = '1 0 0'
    A = 3.14159265E-04 # 2.0 cm (0.02 m) in diameter, A = 1/4 * PI * d^2
    Dh = 2.e-2
    f = 0.01
    Hw = 0.0 # not setting Hw means that Hw is calculated by models, need set 0 for no heat transfer
    length = 1
    n_elems = 50
  [../]
  [./pipe2]
    # geometry
    type = Pipe
    eos = eos
    position = '1.2 0 0'
    orientation = '1 0 0'
    A = 0.785398163e-4 # 1.0 cm (0.01 m) in diameter, A = 1/4 * PI * d^2
    Dh = 1.e-2
    f = 0.01
    Hw = 0.0
    length = 1
    n_elems = 50
  [../]
  [./pump]
    # now no-used but still required parameters, give them some whatever values
    type = IdealPump
    eos = eos
    inputs = 'pipe1(out)'
    outputs = 'pipe2(in)'
    mass_flow_rate = 0.3141159265 # rho * u * A (kg/s)
    Area = 2.624474
    Initial_pressure = 151.7e5
  [../]
  [./inlet_TDV]
    type = TimeDependentVolume
    input = 'pipe1(in)'
    p_bc = 1.0e5
    T_bc = 300.0
    eos = eos
  [../]
  [./outlet_TDV]
    type = TimeDependentVolume
    input = 'pipe2(out)'
    p_bc = 1.e5
    T_bc = 300.0
    eos = eos
  [../]
[]
[Preconditioning]
  # Uncomment one of the lines below to activate one of the blocks...
  # active = 'SMP_PJFNK' 
  # active = 'FDP_PJFNK'
  # active = 'FDP_Newton' 
  # active = 'my_PBP'
  # The definitions of the above-named blocks follow.
  # End preconditioning block
  active = 'SMP_Newton' # SMP_Newton is currently not converging for me, likely an error in computeQpJacobian() somewhere!
  [./SMP_PJFNK]
    type = SMP
    full = true

  #Preconditioned JFNK (default)
  solve_type = 'PJFNK'


  [../]
  [./SMP_Newton]
    type = SMP
    full = true
    petsc_options = -snes
  [../]
  [./FDP_PJFNK]
    # petsc_options_iname = '-mat_fd_type'
    # petsc_options_value = 'ds'
    type = FDP
    full = true

  #Preconditioned JFNK (default)
  solve_type = 'PJFNK'


    petsc_options_iname = '-mat_fd_coloring_err'
    petsc_options_value = 1.e-10
  [../]
  [./FDP_Newton]
    # petsc_options_iname = '-mat_fd_type'
    # petsc_options_value = 'ds'
    type = FDP
    full = true
    petsc_options = -snes
    petsc_options_iname = '-mat_fd_coloring_err'
    petsc_options_value = 1.e-10
  [../]
  [./my_PBP]
    # "Standard" cycle
    # Each "vertical" pair in these vectors gives the (row,col) coordinates of the off-diagonal block
    # Specify *All* off-diagonal couplings (2D)
    type = PBP
    petsc_options = '-snes_mf'
    solve_order = 'rho rhou'
    preconditioner = 'ILU  ILU'
    off_diag_row = 'rho  rhou'
    off_diag_column = 'rhou rho '
  [../]
[]
[Executioner]
  # type = DT2Transient
  # When I tried to run this out to 5000 timesteps with dt=6.25e-4, 
  # it eventually started to bump down the timestep and eventually reached
  # dtmin and the simulation failed, even though the residuals were
  # quite small... (bottoming out around 1.e-6, which is very close to the
  # relative residual tolerance of 1.e-5.)  Scaling the momentum equation does 
  # help with this issue, but the nonlinear convergence does still fail
  # albeit with low residuals (around timestep 3273).  I suppose the FDP
  # finite difference parameter (mat_fd_coloring_err) could also be playing
  # into this issue?
  # With pc_type=lu, we can use dt=1
  # setting time step range
  # time_t = '0      0.1     0.5      1.0     2.0		10'
  # time_dt ='5.e-3  1.1e-2   2.1e-1    2.1e-1   2.1e-1	1.'
  # time_dt ='1.e-2  1.1e-2   2.1e-2    5.1e-2   5.1e-2	1.'
  # 
  # 
  # The CFL condition (|U|*dt/dx <= 1) for this particular problem implies
  # dt <= 2.4e-4.  This is an explicit timestepping limitation, so it would be nice
  # if we could take a larger timestep than this...however in practice I found
  # the Newton solver also had trouble with 2.4e-4.  2.e-4 seems to be OK, though
  # the linear solver does have to work quite hard and it seems like it might not
  # be worth it?  Meanwhile 1.e-4 is "easy".  1.5e-4 seems to be reasonable for
  # later stages of the computation, not sure if it's applicable for the early
  # stages, though...
  # Note: dtmax is also obeyed by the normal Transient Executioner
  # Required parameters for DT2Transient:
  # Target error tolerance, implies desired truncation error, E ~ e_tol*|U2|.
  # Grows timestep by (e_tol*|U2|/|U2-U1|)^(1/p) where U2 is the 2-step solution, 
  # U1 is the 1-step solution.  In case you are only interested in timestepping
  # to steady state, this number could perhaps be quite large.  If e_tol is too large,
  # however, this may lead to too-rapid increase in dt and cause Newton to fail!
  # Max-allowable (relative) error between the 1- and 2-step solutions |U2-U1|/max(|U2|,|U1|).
  # A value of 1.0, for example, would indicate 100% relative error
  # allowed between the U1 and U2 solutions.  The classical "step-doubling" method
  # does not discuss this additional tolerance, it is assumed that the step is always
  # accepted although the timestep may be reduced.  To accept any step no matter what,
  # just set an arbitrary large value here.
  # optional, the maximum amount by which the timestep can increase in a single update
  # 
  # 
  # These options *should* append to any options set in Preconditioning blocks above  
  # Zero pivot options in PETSc 2.3.x:
  # -pc_factor_shift_nonzero: Shift added to diagonal (PCFactorSetShiftNonzero)
  # -pc_factor_shift_nonzero <0>: Shift added to diagonal (PCFactorSetShiftNonzero)
  # -pc_factor_shift_positive_definite: Manteuffel shift applied to diagonal (PCFactorSetShiftPd)
  # -pc_factor_zeropivot <1e-12>: Pivot is considered zero if less than (PCFactorSetZeroPivot)
  # Also consider running with -snes_ksp_ew -mat_mffd_type ds, but be aware
  # that, with -snes_ksp_ew at least, the solver often gives up too easily
  # and cuts the timestep.
  # nl_abs_step_tol = 1e-15
  # use 500 to test the physics
  # num_steps = 500
  # end_time = 5.
  # close Executioner section
  type = RavenExecutioner
  dt = 1.e-1
  dtmin = 1.e-5
  dtmax = 9999
  e_tol = 10.0
  e_max = 99999.
  max_increase = 10
  petsc_options_iname = '-ksp_gmres_restart -pc_type'
  petsc_options_value = '300 lu'
  nl_rel_tol = 1e-6
  nl_abs_tol = 1e-8
  nl_max_its = 10
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
[Output]
  # Turn on performance logging
  exodus = true
  output_initial = true
  output_displaced = true
  perf_log = true
[]
[Controlled]
  control_logic_input = 'ideal_pump_control'
  [./pipe1_Area]
    print_csv = true
    component_name = pipe1
    property_name = Area
    data_type = double
  [../]
  [./pipe1_Dh]
    print_csv = true
    component_name = pipe1
    property_name = Dh
    data_type = double
  [../]
  [./pipe1_Hw]
    print_csv = true
    component_name = pipe1
    property_name = Hw
    data_type = double
  [../]
  [./pipe1_aw]
    print_csv = true
    component_name = pipe1
    property_name = aw
    data_type = double
  [../]
  [./pipe1_f]
    component_name = pipe1
    property_name = f
    data_type = double
  [../]
  [./pipe2_Area]
    component_name = pipe2
    property_name = Area
    data_type = double
  [../]
  [./pipe2_Dh]
    component_name = pipe2
    property_name = Dh
    data_type = double
  [../]
  [./pipe2_Hw]
    component_name = pipe2
    property_name = Hw
    data_type = double
  [../]
  [./pipe2_aw]
    component_name = pipe2
    property_name = aw
    data_type = double
  [../]
  [./pipe2_f]
    component_name = pipe2
    property_name = f
    data_type = double
  [../]
  [./pump_mass_flow_rate]
    print_csv = true
    component_name = pump
    property_name = 'mass_flow_rate'
    data_type = double
  [../]
  [./inlet_TDV_p_bc]
    component_name = 'inlet_TDV'
    property_name = 'p_bc'
    data_type = double
  [../]
  [./inlet_TDV_T_bc]
    component_name = 'inlet_TDV'
    property_name = 'T_bc'
    data_type = double
  [../]
  [./inlet_TDV_void_fraction_bc]
    component_name = 'inlet_TDV'
    property_name = 'void_fraction_bc'
    data_type = double
  [../]
  [./outlet_TDV_p_bc]
    component_name = 'outlet_TDV'
    property_name = 'p_bc'
    data_type = double
  [../]
  [./outlet_TDV_T_bc]
    component_name = 'outlet_TDV'
    property_name = 'T_bc'
    data_type = double
  [../]
  [./outlet_TDV_void_fraction_bc]
    component_name = 'outlet_TDV'
    property_name = 'void_fraction_bc'
    data_type = double
  [../]
[]
[Monitored]
[]
[RavenAuxiliary]
  [./aBoolean]
    data_type =  bool
    initial_value =  False
print_csv = True
  [../]
[./dummy_for_branch]
  data_type =  double
  initial_value =  0.0
  print_csv = True
[../]
[]
[Distributions]
 RNG_seed = 1
 [./zeroToOne]
 type = UniformDistribution
 xMin = 0.0
 xMax = 1.0
 ProbabilityThreshold = 0.1
  
 [../]
[]
