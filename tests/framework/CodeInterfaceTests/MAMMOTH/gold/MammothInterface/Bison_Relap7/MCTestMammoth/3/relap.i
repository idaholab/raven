[GlobalParams]
  # 7 eqn global parameters
  stabilization_entropy_viscosity_Cjump_liquid = 4
  initial_T_vapor = 619
  stabilization_entropy_viscosity_use_jump_vapor = true
  initial_v_vapor = 0
  scaling_factor_2phase = '1e0
1e0 1e0 1e-3
1e0 1e0 1e-3'
  scaling_factor_temperature = 1.e0
  initial_p_liquid = 15.5e6
  stabilization_entropy_viscosity_use_first_order_vf = false
  initial_p_vapor = 15.5e6
  stabilization_entropy_viscosity_Cjump_vapor = 4
  wall_mass_transfer = true
  stabilization_entropy_viscosity_use_first_order_vapor = false
  gravity = '0.0 -9.81 0.0'
  interface_transfer = true
  heat_flux_partitioning_model = linear
  stabilization_entropy_viscosity_use_first_order_liquid = false
  stabilization_entropy_viscosity_use_jump_vf = true
  stabilization_entropy_viscosity_use_jump_liquid = true
  velocity_relaxation = true
  initial_volume_fraction_vapor = 0.01
  initial_v_liquid = 0
  initial_T_liquid = 305.556591398
  phase_interaction = true
  pressure_relaxation = true
  specific_interfacial_area_max_value = 10
  stabilization_entropy_viscosity_Cjump_vf = 4
  explicit_acoustic_impedance = true
  stabilization_type = ENTROPY_VISCOSITY
[]
[FluidProperties]
  [./fp]
    type = IAPWS957EqnFluidProperties
  [../]
[]
[Components]
  [./core]
    # geometry
    A = 1.0731681e-4
    fp = fp
    orientation = '0 1 0'
    Dh = 5.6e-3
    f = 0.01
    heat_transfer_geom = VERTICAL_CYLINDER_BUNDLE_W_CROSS_FLOW
    Hw_liquid = 33000
    Phf = 2.9530971e-2
    Tw_transferred = true
    Hw_vapor = 0
    length = 0.139
    f_interface = 0
    n_elems = 100
    position = '20 0 0'
    type = Pipe
  [../]
  [./inlet]
    volume_fraction_vapor = 0.01
    u_liquid = 4.0
    rho_liquid = 1005.141646742727858
    rho_vapor = 100.027005920197880
    u_vapor = 4.0
    input = 'core(in)'
    type = Inlet
  [../]
  [./outlet]
    p_vapor = 15.5e6
    input = 'core(out)'
    type = Outlet
    p_liquid = 15.5e6
  [../]
[]
[AuxVariables]
  [./pressure_mix]
    order = FIRST
    family = LAGRANGE
  [../]
  [./density_mix]
    order = FIRST
    family = LAGRANGE
  [../]
[]
[AuxKernels]
  [./pm_aux]
    variable = pressure_mix
    a = 'alpha_liquid alpha_vapor'
    b = 'pressure_liquid pressure_vapor'
    type = MixtureQuantityAux
    execute_on = 'timestep_end'
  [../]
  [./density_aux]
    variable = density_mix
    a = 'alpha_liquid alpha_vapor'
    b = 'density_liquid density_vapor'
    type = MixtureQuantityAux
    execute_on = 'timestep_end'
  [../]
[]
[Preconditioning]
  [./SMP_PJFNK]
    full = true
    type = SMP
    line_search = basic
    solve_type = 'PJFNK'
  [../]
[]
[Executioner]
  nl_abs_tol = 1e-6
  nl_max_its = 7
  l_tol = 1e-2
  start_time = 0
  num_steps = 1
  nl_rel_tol = 1e-8
  l_max_its = 50
  dtmin = 1.e-6
  dt = 1e-3
  type = Transient
  [./Quadrature]
    type = TRAP
    order = FIRST
  [../]
[]
[Outputs]
  file_base = out~relap
  csv = true
  [./out_displaced]
    execute_on = 'initial timestep_end'
    use_displaced = true
    type = Exodus
    sequence = false
  [../]
  [./console]
    perf_log = true
    type = Console
  [../]
[]
