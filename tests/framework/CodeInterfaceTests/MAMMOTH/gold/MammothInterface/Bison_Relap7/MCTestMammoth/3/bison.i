[GlobalParams]
  family = LAGRANGE
  disp_z = disp_y
  disp_y = disp_y
  disp_x = disp_x
  density =  10531
  disp_r = disp_x
  order = SECOND
[]
[Problem]
  solution_variables = 'disp_x disp_y temp'
  acceptable_iterations = 10
  reference_residual_variables = 'saved_x saved_y saved_t'
  acceptable_multiplier = 10
  coord_type = RZ
  type = ReferenceResidualProblem
[]
[Mesh]
  clad_mesh_density = customize
  ny_cu = 2
  pellet_mesh_density = customize
  elem_type = QUAD9
  ny_p = 26
  nx_c = 4
  patch_size = 30
  clad_gap_width = 50e-6
  nx_p = 10
  pellet_outer_radius = 4.13e-3
  clad_thickness = 5.7e-4
  clad_top_gap_height = 1.8662e-2
  ny_c = 25
  clad_bot_gap_height = 1.8662e-2
  pellet_height = 1.e-2
  pellet_quantity = 10
  ny_cl = 2
  type = SmearedPelletMesh
  displacements = 'disp_x disp_y'
  top_bot_clad_height = 1.e-3
[]
[Variables]
  [./disp_x]
  [../]
  [./disp_y]
  [../]
  [./temp]
    initial_condition = 293.15
  [../]
[]
[AuxVariables]
  # from RELAP-7
  [./fast_neutron_flux]
    block = 'clad'
  [../]
  [./fast_neutron_fluence]
    block = 'clad'
  [../]
  [./stress_xx]
    # stress aux variables are defined for output; this is a way to get integration point variables to the output file
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_yy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_zz]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./creep_strain_xx]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./creep_strain_yy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./creep_strain_xy]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./creep_strain_hoop]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./vonmises]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./creep_strain_mag]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./plastic_strain_hoop]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./plastic_strain_mag]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./gap_cond]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./burnup]
    order = CONSTANT
    family = MONOMIAL
    initial_condition = 0
  [../]
  [./saved_x]
  [../]
  [./saved_y]
  [../]
  [./saved_t]
  [../]
  [./temperature_vapor]
    initial_condition = 619
  [../]
  [./temperature_liquid]
    initial_condition = 293.15
  [../]
  [./kappa_vapor]
    initial_condition = 0.01
  [../]
  [./kappa_liquid]
    initial_condition = 0.99
  [../]
  [./Hw_vapor]
    initial_condition = 0
  [../]
  [./Hw_liquid]
    initial_condition = 0
  [../]
  [./pressure_mix]
    initial_condition = 15.5e6
  [../]
[]
[Functions]
  [./power_history]
    scale_factor = 1
    xy_data = '0       0
100     0
100.03  1
100.06  0
200     0'
    type = PiecewiseLinear
    format = columns
  [../]
  [./fast_flux_history]
    scale_factor = 0
    xy_data = '0       0
1e8     0'
    type = PiecewiseLinear
    format = columns
  [../]
[]
[SolidMechanics]
  [./solid]
    save_in_disp_r = saved_x
    save_in_disp_z = saved_y
    temp = temp
  [../]
[]
[Kernels]
  [./heat]
    # gradient term in heat conduction equation
    variable = temp
    save_in = saved_t
    type = HeatConduction
  [../]
  [./heat_ie]
    # time term in heat conduction equation
    variable = temp
    save_in = saved_t
    type = HeatConductionTimeDerivative
  [../]
  [./heat_source_fuel]
    # source term in heat conduction equation
    function = power_history
    save_in = saved_t
    value =  1.8661665e11
    variable = temp
    type = HeatSource
    block = 'pellet'
  [../]
[]
[AuxKernels]
  # stress components for output
  # computes stress components for output
  [./fast_neutron_flux]
    variable = fast_neutron_flux
    function = fast_flux_history
    type = FastNeutronFluxAux
    block = 'clad'
    execute_on = timestep_begin
  [../]
  [./fast_neutron_fluence]
    variable = fast_neutron_fluence
    fast_neutron_flux = fast_neutron_flux
    type = FastNeutronFluenceAux
    execute_on = timestep_begin
  [../]
  [./stress_xx]
    variable = stress_xx
    index = 0
    type = MaterialTensorAux
    tensor = stress
    execute_on =  timestep_end
  [../]
  [./stress_yy]
    variable = stress_yy
    index = 1
    type = MaterialTensorAux
    tensor = stress
    execute_on = timestep_end
  [../]
  [./stress_zz]
    variable = stress_zz
    index = 2
    type = MaterialTensorAux
    tensor = stress
    execute_on = timestep_end
  [../]
  [./vonmises]
    variable = vonmises
    execute_on = timestep_end
    type = MaterialTensorAux
    tensor = stress
    quantity = vonmises
  [../]
  [./creep_strain_xx]
    index = 0
    tensor = creep_strain
    variable = creep_strain_xx
    execute_on =  timestep_end
    type = MaterialTensorAux
    block = clad
  [../]
  [./creep_strain_yy]
    index = 1
    tensor = creep_strain
    variable = creep_strain_yy
    execute_on = timestep_end
    type = MaterialTensorAux
    block = clad
  [../]
  [./creep_strain_xy]
    index = 3
    tensor = creep_strain
    variable = creep_strain_xy
    execute_on = timestep_end
    type = MaterialTensorAux
    block = clad
  [../]
  [./creep_strain_hoop]
    index = 2
    tensor = creep_strain
    variable = creep_strain_hoop
    execute_on = timestep_end
    type = MaterialTensorAux
    block = clad
  [../]
  [./creep_strain_mag]
    tensor = creep_strain
    variable = creep_strain_mag
    execute_on = timestep_end
    type = MaterialTensorAux
    block = clad
    quantity = plasticstrainmag
  [../]
  [./plastic_strain_hoop]
    index = 2
    tensor = plastic_strain
    variable = plastic_strain_hoop
    execute_on = timestep_end
    type = MaterialTensorAux
    block = clad
  [../]
  [./plastic_strain_mag]
    tensor = plastic_strain
    variable = plastic_strain_mag
    execute_on = timestep_end
    type = MaterialTensorAux
    block = clad
    quantity = plasticstrainmag
  [../]
  [./conductance]
    variable = gap_cond
    boundary = 10
    property = gap_conductance
    type = MaterialRealAux
  [../]
[]
[Contact]
  [./pellet_clad_mechanical]
    slave = 10
    system = Constraint
    penalty = 1e7
    master = 5
    model = frictionless
    normal_smoothing_distance = 0.1
  [../]
[]
[ThermalContact]
  [./thermal_contact]
    gap_conductivity =  0.1513
    quadrature = true
    master = 5
    variable = temp
    slave = 10
    type = GapHeatTransfer
    normal_smoothing_distance = 0.1
  [../]
[]
[BCs]
  # For RELAP-7
  # pin pellets and clad along axis of symmetry (y)
  # pin clad bottom in the axial direction (y)
  # pin fuel bottom in the axial direction (y)
  # apply coolant pressure on clad outer walls
  [./convective_clad_surface_2phase]
    T_infinity_liquid = temperature_liquid
    kappa_liquid = kappa_liquid
    Hw_liquid = Hw_liquid
    T_infinity_vapor = temperature_vapor
    Hw_vapor = Hw_vapor
    variable = temp
    kappa_vapor = kappa_vapor
    boundary = '2'
    type = CoupledConvectiveFluxTwoPhase
  [../]
  [./no_x_all]
    variable = disp_x
    boundary = 12
    type = DirichletBC
    value = 0.0
  [../]
  [./no_y_clad_bottom]
    variable = disp_y
    boundary = '1'
    type = DirichletBC
    value = 0.0
  [../]
  [./no_y_fuel_bottom]
    variable = disp_y
    boundary = 20
    type = DirichletBC
    value = 0.0
  [../]
  [./pressure_bc_x]
    variable = disp_x
    pressure = pressure_mix
    component = 0
    type = CoupledPressureBC
    boundary = '1 2 3'
  [../]
  [./pressure_bc_y]
    variable = disp_y
    pressure = pressure_mix
    component = 1
    type = CoupledPressureBC
    boundary = '1 2 3'
  [../]
  [./PlenumPressure]
    # apply plenum pressure on clad inner walls and pellet surfaces
    [./plenumPressure]
        save_in_disp_y = saved_y
        volume =  gas_volume
        temperature =  ave_temp_interior
        save_in_disp_x = saved_x
        startup_time = 0
        initial_pressure = 2.0e6
        R = 8.3143
        output =  plenum_pressure
        boundary = 9
        output_initial_moles =  initial_moles
    [../]
  [../]
[]
[Materials]
  # Model for Zircaloy phase transition
  [./density_fuel]
    type = Density
    block = 'pellet'
  [../]
  [./fuel_thermal]
    temp = temp
    initial_porosity = 0.04
    block = 'pellet'
    model = 4
    type = ThermalFuel
    burnup = burnup
  [../]
  [./fuel_mechanical]
    #   volumetric_strain = deltav_v0_swe
    stress_free_temperature = 297
    temp = temp
    poissons_ratio = 0.430979509717
    thermal_expansion = 10e-6
    youngs_modulus = 2.e11
    type = Elastic
    block = 'pellet'
  [../]
  [./density_clad]
    type = Density
    block = clad
    density = 6551.0
  [../]
  [./clad_thermal]
    type = ThermalZry
    temp = temp
    block = clad
  [../]
  [./clad_solid_mechanics]
    stress_free_temperature = 297.0
    model_irradiation_growth = true
    temp = temp
    constitutive_model = combined
    poissons_ratio = 0.3
    thermal_expansion = 5.0e-6
    youngs_modulus = 7.5e10
    type = MechZry
    model_thermal_expansion = false
    block = clad
  [../]
  [./combined]
    relative_tolerance = 1e-3
    type = CombinedCreepPlasticity
    temp = temp
    block = clad
    submodels = 'creep plasticity'
  [../]
  [./creep]
    fast_neutron_fluence = fast_neutron_fluence
    model_thermal_creep_loca = true
    temp = temp
    relative_tolerance = 1e-3
    absolute_tolerance = 1e-5
    fast_neutron_flux = fast_neutron_flux
    type = CreepZryModel
    block = clad
  [../]
  [./plasticity]
    temp = temp
    relative_tolerance = 1e-3
    yield_stress = 500e6
    hardening_constant = 2.5e9
    type = IsotropicPlasticity
    block = clad
  [../]
  [./phase]
    numerical_method = 2
    type = ZrPhase
    temperature = temp
    block = clad
  [../]
[]
[Dampers]
  [./limitT]
    variable = temp
    type = MaxIncrement
    max_increment = 50.0
  [../]
[]
[Executioner]
  # controls for linear iterations
  # controls for nonlinear iterations
  # time control
  nl_abs_tol = 1e-8
  petsc_options_value = 'asm      lu           20              101'
  nl_max_its = 15
  type = Transient
  start_time = 0
  dtmax = 100
  num_steps = 1
  line_search = 'none'
  l_tol = 1e-2
  nl_rel_tol = 1e-10
  solve_type = 'PJFNK'
  petsc_options = '-snes_ksp_ew'
  dtmin = 1e-6
  dt = 1e-3
  petsc_options_iname = '-pc_type -sub_pc_type -pc_asm_overlap -ksp_gmres_restart'
  picard_max_its = 2
  l_max_its = 50
  [./Quadrature]
    order = FIFTH
    side_order = SEVENTH
  [../]
[]
[Postprocessors]
  # average temperature of the cladding interior and all pellet exteriors
  # gas volume
  # area integrated heat flux from the cladding
  # area integrated heat flux from the fuel
  [./ave_temp_interior]
    variable = temp
    execute_on = linear
    boundary = 9
    type = SideAverageValue
  [../]
  [./gas_volume]
    execute_on = linear
    boundary = 9
    type = InternalVolume
  [../]
  [./flux_from_clad]
    variable = temp
    boundary = 5
    diffusivity = thermal_conductivity
    type = SideFluxIntegral
  [../]
  [./flux_from_fuel]
    variable = temp
    boundary = 10
    diffusivity = thermal_conductivity
    type = SideFluxIntegral
  [../]
  [./input_rod_power]
    function = power_history
    type = FunctionValuePostprocessor
  [../]
  [./peak_temp]
    variable = temp
    type = NodalMaxValue
  [../]
  [./peak_clad_temp]
    variable = temp
    boundary = 'clad_outside_right'
    type = NodalMaxValue
  [../]
[]
[Outputs]
  #output_initial = true
  color = false
  file_base = out~bison
  interval = 1
  exodus = true
  csv = true
  [./console]
    output_linear = true
    perf_log = true
    type = Console
    max_rows = 25
  [../]
[]
[Debug]
  show_var_residual_norms = true
  show_var_residual = 'temp'
[]
[MultiApps]
  [./relap]
    max_procs_per_app = 1
    app_type = RELAP7App
    positions = '0 0 0'
    sub_cycling = false
    output_sub_cycles = true
    execute_on = timestep_end
    input_files = relap.i
    type = TransientMultiApp
    detect_steady_state = true
  [../]
[]
[Transfers]
  # to RELAP-7
  # from RELAP-7
  [./clad_surface_temp_to_relap]
    source_variable = temp
    direction = to_multiapp
    displaced_target_mesh = true
    variable = Tw
    fixed_meshes = true
    type = MultiAppNearestNodeTransfer
    multi_app = relap
  [../]
  [./coolant_vapor_to_clad_temp]
    source_variable = temperature_vapor
    direction = from_multiapp
    variable = temperature_vapor
    fixed_meshes = true
    type = MultiAppNearestNodeTransfer
    displaced_source_mesh = true
    multi_app = relap
  [../]
  [./coolant_liquid_to_clad_temp]
    source_variable = temperature_liquid
    direction = from_multiapp
    variable = temperature_liquid
    fixed_meshes = true
    type = MultiAppNearestNodeTransfer
    displaced_source_mesh = true
    multi_app = relap
  [../]
  [./volume_fraction_vapor_to_clad_temp]
    source_variable = alpha_vapor
    direction = from_multiapp
    variable = kappa_vapor
    fixed_meshes = true
    type = MultiAppNearestNodeTransfer
    displaced_source_mesh = true
    multi_app = relap
  [../]
  [./volume_fraction_liquid_to_clad_temp]
    source_variable = alpha_liquid
    direction = from_multiapp
    variable = kappa_liquid
    fixed_meshes = true
    type = MultiAppNearestNodeTransfer
    displaced_source_mesh = true
    multi_app = relap
  [../]
  [./Hw_vapor_from_multiapp]
    source_variable = Hw_vapor
    direction = from_multiapp
    variable = Hw_vapor
    fixed_meshes = true
    type = MultiAppNearestNodeTransfer
    displaced_source_mesh = true
    multi_app = relap
  [../]
  [./Hw_liquid_from_multiapp]
    source_variable = Hw_liquid
    direction = from_multiapp
    variable = Hw_liquid
    fixed_meshes = true
    type = MultiAppNearestNodeTransfer
    displaced_source_mesh = true
    multi_app = relap
  [../]
  [./pressure_mix_from_multiapp]
    source_variable = pressure_mix
    direction = from_multiapp
    variable = pressure_mix
    fixed_meshes = true
    type = MultiAppNearestNodeTransfer
    displaced_source_mesh = true
    multi_app = relap
  [../]
[]
