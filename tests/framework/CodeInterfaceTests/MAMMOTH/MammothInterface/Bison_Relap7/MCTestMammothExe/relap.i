#
# Flow through one core channel
#

# Transfer into Tw


[GlobalParams]
  stabilization_type = ENTROPY_VISCOSITY
  stabilization_entropy_viscosity_use_first_order_liquid = false
  stabilization_entropy_viscosity_use_first_order_vapor = false
  stabilization_entropy_viscosity_use_first_order_vf = false
  stabilization_entropy_viscosity_use_jump_liquid = true
  stabilization_entropy_viscosity_use_jump_vapor = true
  stabilization_entropy_viscosity_use_jump_vf = true
  stabilization_entropy_viscosity_Cjump_liquid = 4
  stabilization_entropy_viscosity_Cjump_vapor = 4
  stabilization_entropy_viscosity_Cjump_vf = 4

  initial_p_liquid = 15.5e6
  initial_p_vapor  = 15.5e6
  initial_v_liquid = 0
  initial_v_vapor  = 0
  initial_T_liquid = 293.15
  initial_T_vapor  = 619
  initial_volume_fraction_vapor = 0.01

  # 7 eqn global parameters
  phase_interaction = true
  pressure_relaxation = true
  velocity_relaxation = true
  interface_transfer = true
  wall_mass_transfer = true

  specific_interfacial_area_max_value = 10
  explicit_acoustic_impedance = true
  heat_flux_partitioning_model = linear

  scaling_factor_2phase = '1e0
                           1e0 1e0 1e-3
                           1e0 1e0 1e-3'
  scaling_factor_temperature = 1.e0

  gravity = '0.0 -9.81 0.0'
[]

[FluidProperties]
  [./fp]
    type = IAPWS957EqnFluidProperties
  [../]
[]

[Components]
  [./core]
    type = Pipe
    # geometry
    position = '20 0 0'
    orientation = '0 1 0'
    heat_transfer_geom = VERTICAL_CYLINDER_BUNDLE_W_CROSS_FLOW
    length = 0.139
    n_elems = 100

    A   =  1.0731681e-4
    Dh  =  5.6e-3
    Phf =  2.9530971e-2

    f = 0.01
    f_interface = 0
    Hw_liquid = 33000
    Hw_vapor = 0

    Tw_transferred = true

    fp = fp
  [../]

  [./inlet]
    type = Inlet
    input = 'core(in)'

    rho_liquid = 1005.141646742727858
    u_liquid   =    4.0
    rho_vapor  =  100.027005920197880
    u_vapor    =    4.0
    volume_fraction_vapor =  0.01
  [../]

  [./outlet]
    type = Outlet
    input = 'core(out)'

    p_liquid = 15.5e6
    p_vapor = 15.5e6
  [../]
[]

[AuxVariables]
  [./pressure_mix]
    family = LAGRANGE
    order = FIRST
  [../]

  [./density_mix]
    family = LAGRANGE
    order = FIRST
  [../]
[]

[AuxKernels]
  [./pm_aux]
    type = MixtureQuantityAux
    variable = pressure_mix
    a = 'alpha_liquid alpha_vapor'
    b = 'pressure_liquid pressure_vapor'
    execute_on = 'timestep_end'
  [../]

  [./density_aux]
    type = MixtureQuantityAux
    variable = density_mix
    a = 'alpha_liquid alpha_vapor'
    b = 'density_liquid density_vapor'
    execute_on = 'timestep_end'
  [../]
[]

[Preconditioning]
  [./SMP_PJFNK]
    type = SMP
    full = true
    solve_type = 'PJFNK'
    line_search = basic
  [../]
[]

[Executioner]
  type = Transient

  dtmin = 1.e-6
  dt = 1e-3

  nl_rel_tol = 1e-8
  nl_abs_tol = 1e-6
  nl_max_its = 7

  l_tol = 1e-2
  l_max_its = 50

  start_time = 0
  num_steps = 1

  [./Quadrature]
    type = TRAP
    order = FIRST
  [../]
[]

[Outputs]
  [./out_displaced]
    type = Exodus
    use_displaced = true
    execute_on = 'initial timestep_end'
    sequence = false
  [../]

  [./console]
    type = Console
    perf_log = true
  [../]
[]
