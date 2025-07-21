[Mesh]
  distribution = serial
  uniform_refine = 0
  file = QuarterPin_FuelGapCladWater_QUAD8_Fine.e
  second_order = true
[]
[GlobalParams]
  isotopes = 'pseudo'
  grid_variables = 'burnup Temp'
  densities = 1.0
  forDiffusion = false
  dbgmat = true
  allowVacuum = true
  isMeter = true
  forTransient = false
  grid_names = 'Burnup Tfuel'
  forAdjoint = false
  plus = true
  ngroup = 8
[]
[TransportSystems]
  equation_type = eigenvalue
  G = 8
  particle = neutron
  ReflectingBoundary = '1001 1002 1003 1004'
  [./sn]
    scheme = SAAF-CFEM-SN
    AQtype = Level-Symmetric
    order = FIRST
    family = LAGRANGE
    AQorder = 2
  [../]
[]
[Depletion]
  # rated power in MW / m
  # area of the fuel in m^2
  # density of the fuel in kg / m^3
  # fuel composition
  power_modulating_function = PowerModulator
  isotopes = 'U235 U238 O16'
  fuel_volume = 4138476.487714
  rated_power = 0.005
  burnup_unit = MWd/kg
  fuel_density = 10480.0
  transport_system = sn
  weight_percentages = '3.922 84.222 11.856'
[]
[AuxVariables]
  [./Temp]
    initial_condition = 642.014950816
  [../]
  [./burnup_fima]
    order = CONSTANT
    family = MONOMIAL
    initial_condition = 0.0
  [../]
  [./fission_rate]
    order = CONSTANT
    family = MONOMIAL
    initial_condition = 0.0
  [../]
  [./total_reactor_power_density_watt]
    order = CONSTANT
    family = MONOMIAL
    initial_condition = 0.0
  [../]
[]
[AuxKernels]
  #
  # Setting the fuel temperature
  #
  [./SetTCool]
    variable = Temp
    type = ConstantAux
    value = 622.0
    block = water
  [../]
  [./burnup_fima_aux]
    # in unit of J/cm
    variable_to_integrate = total_reactor_power_density
    burnup_unit = FIMA
    burnup_unit_converter = burnup_converter_uo
    variable = burnup_fima
    type = PowerDensityTimeIntegrator
    block = 'ring1 ring2 ring3 ring4 ring5
ring6 ring7 ring8 ring9 ring10
ring11 ring12 ring13 ring14 ring15
ring16 ring17 ring18 ring19 ring20
gap clad water'
  [../]
  [./fission_rate_aux]
    scale_factor = power_scaling
    scalar_flux = 'flux_moment_g0_L0_M0
flux_moment_g1_L0_M0
flux_moment_g2_L0_M0
flux_moment_g3_L0_M0
flux_moment_g4_L0_M0
flux_moment_g5_L0_M0
flux_moment_g6_L0_M0
flux_moment_g7_L0_M0'
    variable = fission_rate
    cross_section = sigma_fission
    type = VectorReactionRate
    block = 'ring1 ring2 ring3 ring4 ring5
ring6 ring7 ring8 ring9 ring10
ring11 ring12 ring13 ring14 ring15
ring16 ring17 ring18 ring19 ring20'
  [../]
  [./total_reactor_power_density_watt_aux]
    variable = total_reactor_power_density_watt
    source_variable = total_reactor_power_density
    type = ScaleAux
    multiplier = 1000
  [../]
[]
[YAKXSLibraries]
  [./RingLib1]
    library_type = MultigroupLibrary
    library_name = ring1
    library_file = M1_2D.xml
    type = BaseLibObject
    debug = 3
  [../]
  [./RingLib2]
    library_type = MultigroupLibrary
    library_name = ring2
    library_file = M2_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib3]
    library_type = MultigroupLibrary
    library_name = ring3
    library_file = M3_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib4]
    library_type = MultigroupLibrary
    library_name = ring4
    library_file = M4_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib5]
    library_type = MultigroupLibrary
    library_name = ring5
    library_file = M5_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib6]
    library_type = MultigroupLibrary
    library_name = ring6
    library_file = M6_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib7]
    library_type = MultigroupLibrary
    library_name = ring7
    library_file = M7_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib8]
    library_type = MultigroupLibrary
    library_name = ring8
    library_file = M8_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib9]
    library_type = MultigroupLibrary
    library_name = ring9
    library_file = M9_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib10]
    library_type = MultigroupLibrary
    library_name = ring10
    library_file = M10_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib11]
    library_type = MultigroupLibrary
    library_name = ring11
    library_file = M11_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib12]
    library_type = MultigroupLibrary
    library_name = ring12
    library_file = M12_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib13]
    library_type = MultigroupLibrary
    library_name = ring13
    library_file = M13_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib14]
    library_type = MultigroupLibrary
    library_name = ring14
    library_file = M14_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib15]
    library_type = MultigroupLibrary
    library_name = ring15
    library_file = M15_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib16]
    library_type = MultigroupLibrary
    library_name = ring16
    library_file = M16_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib17]
    library_type = MultigroupLibrary
    library_name = ring17
    library_file = M17_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib18]
    library_type = MultigroupLibrary
    library_name = ring18
    library_file = M18_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib19]
    library_type = MultigroupLibrary
    library_name = ring19
    library_file = M19_2D.xml
    type = BaseLibObject
  [../]
  [./RingLib20]
    library_type = MultigroupLibrary
    library_name = ring20
    library_file = M20_2D.xml
    type = BaseLibObject
  [../]
  [./GapLib]
    library_type = MultigroupLibrary
    library_name = gap
    library_file = M21_2D.xml
    type = BaseLibObject
  [../]
  [./CladLib]
    library_type = MultigroupLibrary
    library_name = clad
    library_file = M22_2D.xml
    type = BaseLibObject
  [../]
  [./WaterLib]
    library_type = MultigroupLibrary
    library_name = water
    library_file = M23_2D.xml
    type = BaseLibObject
  [../]
[]
[Materials]
  [./mat1]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 1
    block = ring1
    MGLibObject = RingLib1
  [../]
  [./mat2]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 2
    block = ring2
    MGLibObject = RingLib2
  [../]
  [./mat3]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 3
    block = ring3
    MGLibObject = RingLib3
  [../]
  [./mat4]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 4
    block = ring4
    MGLibObject = RingLib4
  [../]
  [./mat5]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 5
    block = ring5
    MGLibObject = RingLib5
  [../]
  [./mat6]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 6
    block = ring6
    MGLibObject = RingLib6
  [../]
  [./mat7]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 7
    block = ring7
    MGLibObject = RingLib7
  [../]
  [./mat8]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 8
    block = ring8
    MGLibObject = RingLib8
  [../]
  [./mat9]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 9
    block = ring9
    MGLibObject = RingLib9
  [../]
  [./mat10]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 10
    block = ring10
    MGLibObject = RingLib10
  [../]
  [./mat11]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 11
    block = ring11
    MGLibObject = RingLib11
  [../]
  [./mat12]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 12
    block = ring12
    MGLibObject = RingLib12
  [../]
  [./mat13]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 13
    block = ring13
    MGLibObject = RingLib13
  [../]
  [./mat14]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 14
    block = ring14
    MGLibObject = RingLib14
  [../]
  [./mat15]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 15
    block = ring15
    MGLibObject = RingLib15
  [../]
  [./mat16]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 16
    block = ring16
    MGLibObject = RingLib16
  [../]
  [./mat17]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 17
    block = ring17
    MGLibObject = RingLib17
  [../]
  [./mat18]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 18
    block = ring18
    MGLibObject = RingLib18
  [../]
  [./mat19]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 19
    block = ring19
    MGLibObject = RingLib19
  [../]
  [./mat20]
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 20
    block = ring20
    MGLibObject = RingLib20
  [../]
  [./mat21]
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 21
    block = gap
    MGLibObject = GapLib
  [../]
  [./mat22]
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 22
    block = clad
    MGLibObject = CladLib
  [../]
  [./mat23]
    type = CoupledFeedbackNeutronicsMaterial
    material_id = 23
    block = water
    MGLibObject = WaterLib
  [../]
[]
[Functions]
  [./PowerModulator]
    y = '0 1.0 1.0'
    x = '0 43200.0 30240000'
    type = PiecewiseLinear
  [../]
[]
[Executioner]
  # general PetSc settings
  # in unit of days
  # extra number of free power iterations for solving time 0
  # number of free power iterations for each individual eigen solve
  # absolute convergence on residual for each individual eigen solve
  petsc_options_value = 'hypre boomeramg 0.7 100'
  type = Depletion
  solve_type = 'PJFNK'
  l_max_its = 200
  source_abs_tol = 1e-7
  petsc_options_iname = '-pc_type -pc_hypre_type -pc_hypre_boomeramg_strong_threshold -ksp_gmres_restart '
  picard_max_its = 10
  burnup = ' 0.0 43200 86400 172800 864000 1728000 2592000 3456000 4320000 6480000 8640000 12960000 17280000 21600000 25920000 30240000'
  initial_free_power_iterations = 4
[]
[MultiApps]
  [./sub]
    app_type = BisonApp
    input_files = bison_2way_fuel_clad_elastic.i
    type = TransientMultiApp
    positions = '0.0 0.0 0.0'
  [../]
[]
[Transfers]
  [./tosub]
    variable = burnup
    source_variable = burnup_fima
    direction = to_multiapp
    type = MultiAppInterpolationTransfer
    multi_app = sub
  [../]
  [./tosub2]
    variable = fission_rate
    source_variable = fission_rate
    direction = to_multiapp
    type = MultiAppInterpolationTransfer
    multi_app = sub
  [../]
  [./tosub3]
    variable = power_density
    source_variable = total_reactor_power_density_watt
    direction = to_multiapp
    type = MultiAppInterpolationTransfer
    multi_app = sub
  [../]
  [./fromsub]
    variable = Temp
    source_variable = temp
    direction = from_multiapp
    type = MultiAppInterpolationTransfer
    multi_app = sub
  [../]
[]
[Postprocessors]
  [./FuelArea]
    type = VolumePostprocessor
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10
ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
  [../]
  [./burnup_MWdkg_pp]
    variable = burnup
    type = ElementAverageValue
  [../]
  [./burnup_FIMA_pp]
    variable = burnup_fima
    type = ElementAverageValue
  [../]
  [./TotalReactorPower]
    variable = total_reactor_power_density
    type = ElementIntegralVariablePostprocessor
  [../]
[]
[Outputs]
  file_base = out~quarter_pin_with_fuel
  interval = 1
  csv = true
  [./console]
    perf_log = true
    max_rows = 25
    type = Console
    linear_residuals = true
  [../]
  [./exodus]
    file_base = quarter_pin_with_fuel
    type = Exodus
  [../]
[]
