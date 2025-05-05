[GlobalParams]
  density = 10531     # UO2 with 4% porosity
  disp_x = disp_x
  disp_y = disp_y
  disp_r = disp_x
  disp_z = disp_y
  order  = SECOND
  family = LAGRANGE
[]

# Specify coordinate system type
[Problem]
  coord_type = RZ
  type = ReferenceResidualProblem
  solution_variables = 'disp_x disp_y temp'
  reference_residual_variables = 'saved_x saved_y saved_t'
  acceptable_iterations = 10
  acceptable_multiplier = 10
[]

[Mesh]
  type = SmearedPelletMesh
  clad_mesh_density = customize
  pellet_mesh_density = customize
  nx_p = 10
  ny_p = 26
  ny_cu = 2
  ny_c = 25
  ny_cl = 2
  nx_c = 4
  clad_thickness = 5.7e-4
  pellet_outer_radius = 4.13e-3
  pellet_quantity = 10
  pellet_height = 1.e-2
  clad_top_gap_height = 1.8662e-2
  clad_bot_gap_height = 1.8662e-2
  clad_gap_width = 50e-6
  top_bot_clad_height = 1.e-3
  elem_type = QUAD9
  displacements = 'disp_x disp_y'
  patch_size = 30
[]

# Define dependent variables and initial conditions
[Variables]
  [./disp_x]
  [../]

  [./disp_y]
  [../]

  [./temp]
    initial_condition = 293.15
  [../]

[]

# Define auxillary variables, element order and shape function family
[AuxVariables]
  [./fast_neutron_flux]
    block = 'clad'
  [../]

  [./fast_neutron_fluence]
    block = 'clad'
  [../]

  [./stress_xx]      # stress aux variables are defined for output; this is a way to get integration point variables to the output file
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

  # from RELAP-7
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

# Define functions to control power and boundary conditions
[Functions]
  [./power_history]
    type = PiecewiseLinear
    format = columns
    scale_factor = 1
    xy_data = '0       0
               100     0
               100.03  1
               100.06  0
               200     0'
  [../]
  [./fast_flux_history]
    type = PiecewiseLinear
    format = columns
    scale_factor = 0
    xy_data = '0       0
               1e8     0'
  [../]
[]

# Specify that we need solid mechanics (divergence of stress)
[SolidMechanics]
  [./solid]
    temp = temp
    save_in_disp_r = saved_x
    save_in_disp_z = saved_y
  [../]
[]

# Define kernels for the various terms in the PDE system
[Kernels]
  [./heat]         # gradient term in heat conduction equation
    type = HeatConduction
    variable = temp
    save_in = saved_t
  [../]

 [./heat_ie]       # time term in heat conduction equation
    type = HeatConductionTimeDerivative
    variable = temp
    save_in = saved_t
  [../]

  [./heat_source_fuel]  # source term in heat conduction equation
     type = HeatSource
     variable = temp
     block = 'pellet'
     function = power_history
     value = 1.8661665e11   # 36.e6 W / (Pi * 0.00413m^2 * 3.6m)
     save_in = saved_t
  [../]
 []

# Define auxilliary kernels for each of the aux variables
[AuxKernels]
  [./fast_neutron_flux]
    type = FastNeutronFluxAux
    variable = fast_neutron_flux
    block = 'clad'
    function = fast_flux_history
    execute_on = timestep_begin
  [../]

  [./fast_neutron_fluence]
    type = FastNeutronFluenceAux
    variable = fast_neutron_fluence
    fast_neutron_flux = fast_neutron_flux
    execute_on = timestep_begin
  [../]

  # stress components for output
  [./stress_xx]
    type = MaterialTensorAux
    tensor = stress
    variable = stress_xx
    index = 0
    execute_on = timestep_end     # for efficiency, only compute at the end of a timestep
  [../]

  [./stress_yy]
    type = MaterialTensorAux
    tensor = stress
    variable = stress_yy
    index = 1
    execute_on = timestep_end
  [../]

  [./stress_zz]
    type = MaterialTensorAux
    tensor = stress
    variable = stress_zz
    index = 2
    execute_on = timestep_end
  [../]

  [./vonmises]
    type = MaterialTensorAux
    tensor = stress
    variable = vonmises
    quantity = vonmises
    execute_on = timestep_end
  [../]

  # computes stress components for output
  [./creep_strain_xx]
    type = MaterialTensorAux
    tensor = creep_strain
    variable = creep_strain_xx
    block = clad
    index = 0
    execute_on = timestep_end     # for efficiency, only compute at the end of a timestep
  [../]

  [./creep_strain_yy]
    type = MaterialTensorAux
    tensor = creep_strain
    variable = creep_strain_yy
    block = clad
    index = 1
    execute_on = timestep_end
  [../]

  [./creep_strain_xy]
    type = MaterialTensorAux
    tensor = creep_strain
    variable = creep_strain_xy
    block = clad
    index = 3
    execute_on = timestep_end
  [../]

  [./creep_strain_hoop]
    type = MaterialTensorAux
    tensor = creep_strain
    variable = creep_strain_hoop
    block = clad
    index = 2
    execute_on = timestep_end
  [../]

  [./creep_strain_mag]
    type = MaterialTensorAux
    tensor = creep_strain
    variable = creep_strain_mag
    block = clad
    quantity = plasticstrainmag
    execute_on = timestep_end
  [../]

  [./plastic_strain_hoop]
    type = MaterialTensorAux
    tensor = plastic_strain
    variable = plastic_strain_hoop
    block = clad
    index = 2
    execute_on = timestep_end
  [../]

  [./plastic_strain_mag]
    type = MaterialTensorAux
    tensor = plastic_strain
    variable = plastic_strain_mag
    block = clad
    quantity = plasticstrainmag
    execute_on = timestep_end
  [../]

  [./conductance]
    type = MaterialRealAux
    property = gap_conductance
    variable = gap_cond
    boundary = 10
  [../]
[]

# Define mechanical contact between the fuel (sideset=10) and the clad (sideset=5)
[Contact]
  [./pellet_clad_mechanical]
    master = 5
    slave = 10
    penalty = 1e7
    model = frictionless
    normal_smoothing_distance = 0.1
    system = Constraint
  [../]
[]

# Define thermal contact between the fuel (sideset=10) and the clad (sideset=5)
[ThermalContact]
  [./thermal_contact]
    type = GapHeatTransfer
    variable = temp
    master = 5
    slave = 10
    gap_conductivity = 0.1513 #nominal value for thermal conductivity of He in W/m-K
    quadrature = true
    normal_smoothing_distance = 0.1
  [../]
[]

# Define boundary conditions
[BCs]
  # For RELAP-7
  [./convective_clad_surface_2phase]
    type = CoupledConvectiveFluxTwoPhase
    boundary = '2'
    variable = temp

    T_infinity_vapor = temperature_vapor
    kappa_vapor = kappa_vapor
    Hw_vapor = Hw_vapor
    T_infinity_liquid = temperature_liquid
    kappa_liquid = kappa_liquid
    Hw_liquid = Hw_liquid
  [../]

  # pin pellets and clad along axis of symmetry (y)
  [./no_x_all]
    type = DirichletBC
    variable = disp_x
    boundary = 12
    value = 0.0
  [../]

  # pin clad bottom in the axial direction (y)
  [./no_y_clad_bottom]
    type = DirichletBC
    variable = disp_y
    boundary = '1'
    value = 0.0
  [../]

  # pin fuel bottom in the axial direction (y)
  [./no_y_fuel_bottom]
    type = DirichletBC
    variable = disp_y
    boundary = 20
    value = 0.0
  [../]

  # apply coolant pressure on clad outer walls
  [./pressure_bc_x]
    type = CoupledPressureBC
    variable = disp_x
    pressure = pressure_mix
    component = 0
    boundary = '1 2 3'
  [../]
  [./pressure_bc_y]
    type = CoupledPressureBC
    variable = disp_y
    pressure = pressure_mix
    component = 1
    boundary = '1 2 3'
  [../]

  [./PlenumPressure]
    # apply plenum pressure on clad inner walls and pellet surfaces
    [./plenumPressure]
      boundary = 9
      initial_pressure = 2.0e6
      startup_time = 0
      R = 8.3143
      output_initial_moles = initial_moles       # coupling to post processor to get inital fill gas mass
      temperature = ave_temp_interior            # coupling to post processor to get gas temperature approximation
      volume = gas_volume                        # coupling to post processor to get gas volume
      output = plenum_pressure                   # coupling to post processor to output plenum/gap pressure
      save_in_disp_x = saved_x
      save_in_disp_y = saved_y
    [../]
  [../]
[]

# Define material behavior models and input material property data
[Materials]

  [./density_fuel]
    type = Density
    block = 'pellet'
  [../]
  [./fuel_thermal]
    type = ThermalFuel
    block = 'pellet'
    temp = temp
    burnup = burnup
    initial_porosity = 0.04
    model = 4
  [../]
  [./fuel_mechanical]
    type = Elastic
    block = 'pellet'
    temp = temp
    youngs_modulus = 2.e11
    poissons_ratio = .345
    thermal_expansion = 10e-6
    stress_free_temperature = 297
 #   volumetric_strain = deltav_v0_swe
  [../]

  [./density_clad]
    type = Density
    block = clad
    density = 6551.0
  [../]
  [./clad_thermal]
    type = ThermalZry
    block = clad
    temp = temp
  [../]
  [./clad_solid_mechanics]
    type = MechZry
    block = clad
    temp = temp
    youngs_modulus = 7.5e10
    poissons_ratio = 0.3
    thermal_expansion = 5.0e-6
    stress_free_temperature = 297.0
    constitutive_model = combined
    model_irradiation_growth = true
    model_thermal_expansion = false
  [../]
  [./combined]
    type = CombinedCreepPlasticity
    block = clad
    temp = temp
    submodels = 'creep plasticity'
    relative_tolerance = 1e-3
  [../]
  [./creep]
    type = CreepZryModel
    block = clad
    temp = temp
    fast_neutron_flux = fast_neutron_flux
    fast_neutron_fluence = fast_neutron_fluence
    model_thermal_creep_loca = true
    relative_tolerance = 1e-3
    absolute_tolerance = 1e-5
  [../]
  [./plasticity]
    type = IsotropicPlasticity
    block = clad
    temp = temp
    yield_stress = 500e6
    hardening_constant = 2.5e9
    relative_tolerance = 1e-3
  [../]
  # Model for Zircaloy phase transition
  [./phase]
    type = ZrPhase
    block = clad
    temperature = temp
    numerical_method = 2
  [../]
[]

[Dampers]
  [./limitT]
    type = MaxIncrement
    max_increment = 50.0
    variable = temp
  [../]
[]

[Executioner]
  type = Transient

  solve_type = 'PJFNK'

  line_search = 'none'

  petsc_options_iname = '-pc_type -sub_pc_type -pc_asm_overlap -ksp_gmres_restart'
  petsc_options_value = 'asm      lu           20              101'
  petsc_options = '-snes_ksp_ew'

  # controls for linear iterations
  l_max_its = 50
  l_tol = 1e-2

  # controls for nonlinear iterations
  nl_max_its = 15
  nl_rel_tol = 1e-10
  nl_abs_tol = 1e-8

  # time control
  start_time = 0
  num_steps = 1
  dt = 1e-3
  dtmax = 100
  dtmin = 1e-6
  picard_max_its = 2

  [./Quadrature]
    order = FIFTH
    side_order = SEVENTH
  [../]
[]

[Postprocessors]
  # average temperature of the cladding interior and all pellet exteriors
  [./ave_temp_interior]
     type = SideAverageValue
     boundary = 9
     variable = temp
     execute_on = linear
  [../]

  # gas volume
  [./gas_volume]
    type = InternalVolume
    boundary = 9
    execute_on = linear
  [../]

  # area integrated heat flux from the cladding
  [./flux_from_clad]
    type = SideFluxIntegral
    variable = temp
    boundary = 5
    diffusivity = thermal_conductivity
  [../]

  # area integrated heat flux from the fuel
  [./flux_from_fuel]
    type = SideFluxIntegral
    variable = temp
    boundary = 10
    diffusivity = thermal_conductivity
  [../]

  [./input_rod_power]
    type = FunctionValuePostprocessor
    function = power_history
  [../]

  [./peak_temp]
    type = NodalMaxValue
    variable = temp
  [../]

  [./peak_clad_temp]
    type = NodalMaxValue
    variable = temp
    boundary = 'clad_outside_right'
  [../]
[]

# Define output file(s)
[Outputs]
  interval = 1
  #output_initial = true
  csv = true
  exodus = true
  color = false
  [./console]
    type = Console
    perf_log = true
    output_linear = true
    max_rows = 25
  [../]
[]

[Debug]
  show_var_residual = 'temp'
  show_var_residual_norms = true
[]

[MultiApps]
  [./relap]
    type = TransientMultiApp
    app_type = RELAP7App
    execute_on = timestep_end
    max_procs_per_app = 1
    sub_cycling = false
    output_sub_cycles = true
    detect_steady_state = true
    positions = '0 0 0'
    input_files = relap.i
  [../]
[]

[Transfers]
  # to RELAP-7
  [./clad_surface_temp_to_relap]
    type = MultiAppNearestNodeTransfer
    direction = to_multiapp
    multi_app = relap
    source_variable = temp
    variable = Tw
    fixed_meshes = true
    displaced_target_mesh = true
  [../]

  # from RELAP-7
  [./coolant_vapor_to_clad_temp]
    type = MultiAppNearestNodeTransfer
    direction = from_multiapp
    multi_app = relap
    source_variable = temperature_vapor
    variable = temperature_vapor
    fixed_meshes = true
    displaced_source_mesh = true
  [../]

  [./coolant_liquid_to_clad_temp]
    type = MultiAppNearestNodeTransfer
    direction = from_multiapp
    multi_app = relap
    source_variable = temperature_liquid
    variable = temperature_liquid
    fixed_meshes = true
    displaced_source_mesh = true
  [../]

  [./volume_fraction_vapor_to_clad_temp]
    type = MultiAppNearestNodeTransfer
    direction = from_multiapp
    multi_app = relap
    source_variable = alpha_vapor
    variable = kappa_vapor
    fixed_meshes = true
    displaced_source_mesh = true
  [../]

  [./volume_fraction_liquid_to_clad_temp]
    type = MultiAppNearestNodeTransfer
    direction = from_multiapp
    multi_app = relap
    source_variable = alpha_liquid
    variable = kappa_liquid
    fixed_meshes = true
    displaced_source_mesh = true
  [../]

  [./Hw_vapor_from_multiapp]
    type = MultiAppNearestNodeTransfer
    direction = from_multiapp
    multi_app = relap
    source_variable = Hw_vapor
    variable = Hw_vapor
    fixed_meshes = true
    displaced_source_mesh = true
  [../]

  [./Hw_liquid_from_multiapp]
    type = MultiAppNearestNodeTransfer
    direction = from_multiapp
    multi_app = relap
    source_variable = Hw_liquid
    variable = Hw_liquid
    fixed_meshes = true
    displaced_source_mesh = true
  [../]

  [./pressure_mix_from_multiapp]
    type = MultiAppNearestNodeTransfer
    direction = from_multiapp
    multi_app = relap
    source_variable = pressure_mix
    variable = pressure_mix
    fixed_meshes = true
    displaced_source_mesh = true
  [../]
[]
