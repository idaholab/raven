[Mesh]
  distribution = serial
  uniform_refine = 0
  file = QuarterPin_FuelGapCladWater_QUAD8_Fine.e
  second_order = true
[]
[GlobalParams]
  isotopes = 'pseudo'
  grid_variables = 'fuelburnup Temp'
  densities = 1.0
  forDiffusion = false
  dbgmat = true
  burnup_grid = 1
  allowVacuum = true
  isMeter = true
  forTransient = false
  grid_names = 'Burnup Tfuel'
  forAdjoint = false
  plus = true
  ngroup = 8
[]
[RattleSnakeParams]
  eigenvalue = true
  p0aux2mat = true
  AQtype = Level-Symmetric
  forceflux = true
  G = 8
  NA =  2
  hide_higher_flux_moment = 0
  AQorder = 2
  ReflectingBoundary = '1001 1002 1003 1004'
  depletion =  true
  transient = 0
  calculation_type = SAAF
  n_delay_groups = 0
  order = SECOND
  hide_angular_flux = true
  verbose = 3
[]
[MacroDepletion]
  ## in unit of W/cm for the pin
  ### pin volume
  calculation_type = SAAF
  power_modulating_function = PowerModulator
  rated_power = 4800.0
  G = 8
  family = MONOMIAL
  add_fission_rate = true
  burnup_volume = 1.0
  order = CONSTANT
  block = 'ring1 ring2 ring3 ring4 ring5
ring6 ring7 ring8 ring9 ring10
ring11 ring12 ring13 ring14 ring15
ring16 ring17 ring18 ring19 ring20'
  verbose = 2
[]
[AuxVariables]
  [./Temp]
    initial_condition = 625.435977887
  [../]
  [./fuelburnup]
    order = CONSTANT
    family = MONOMIAL
    initial_condition = 0.0
  [../]
  [./BISONBurnup]
    order = CONSTANT
    family = MONOMIAL
    initial_condition = 0.0
  [../]
  [./BISONFissionRate]
    order = CONSTANT
    family = MONOMIAL
    initial_condition = 0.0
  [../]
  [./BISONPowerDensity]
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
  [./CalcBurnup]
    # in unit of J/cm
    variable_to_integrate = power_density
    burnup_unit = MWdkg
    burnup_unit_converter = BurnupConverter
    time_coef = 0.000011574
    variable = fuelburnup
    type = PowerDensityTimeIntegrator
    block = 'ring1 ring2 ring3 ring4 ring5
ring6 ring7 ring8 ring9 ring10
ring11 ring12 ring13 ring14 ring15
ring16 ring17 ring18 ring19 ring20
gap clad water'
  [../]
  [./CalcBISONBurnup]
    # in unit of J/cm
    variable_to_integrate = power_density
    burnup_unit = FIMA
    burnup_unit_converter = BurnupConverter
    time_coef = 0.000011574
    variable = BISONBurnup
    type = PowerDensityTimeIntegrator
    block = 'ring1 ring2 ring3 ring4 ring5
ring6 ring7 ring8 ring9 ring10
ring11 ring12 ring13 ring14 ring15
ring16 ring17 ring18 ring19 ring20
gap clad water'
  [../]
  [./CalcBISONPowerDensity]
    pp_scale_factor = power_scaling
    scalar_flux = 'flux_moment_g0_L0_M0
flux_moment_g1_L0_M0
flux_moment_g2_L0_M0
flux_moment_g3_L0_M0
flux_moment_g4_L0_M0
flux_moment_g5_L0_M0
flux_moment_g6_L0_M0
flux_moment_g7_L0_M0'
    sigma_xs = kappa_sigma_fission
    variable = BISONPowerDensity
    type = ScaledVectorReactionRate
    block = 'ring1 ring2 ring3 ring4 ring5
ring6 ring7 ring8 ring9 ring10
ring11 ring12 ring13 ring14 ring15
ring16 ring17 ring18 ring19 ring20'
  [../]
  [./CalcBISONFissionRate]
    scale_factor = 3.0445941708e+10
    pp_scale_factor = power_scaling
    scalar_flux = 'flux_moment_g0_L0_M0
flux_moment_g1_L0_M0
flux_moment_g2_L0_M0
flux_moment_g3_L0_M0
flux_moment_g4_L0_M0
flux_moment_g5_L0_M0
flux_moment_g6_L0_M0
flux_moment_g7_L0_M0'
    sigma_xs = kappa_sigma_fission
    variable = BISONFissionRate
    type = ScaledVectorReactionRate
    block = 'ring1 ring2 ring3 ring4 ring5
ring6 ring7 ring8 ring9 ring10
ring11 ring12 ring13 ring14 ring15
ring16 ring17 ring18 ring19 ring20'
  [../]
[]
[UserObjects]
  [./BurnupConverter]
    heavy_metal_isotopes = 'U235 U238'
    density_UO2 = 10480.0
    weight_percentages = '4.45 95.55'
    type = BurnupConverterObject
    power_density = 420.248
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
    MatID = 1
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring1
    MGLibObject = RingLib1
  [../]
  [./mat2]
    MatID = 2
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring2
    MGLibObject = RingLib2
  [../]
  [./mat3]
    MatID = 3
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring3
    MGLibObject = RingLib3
  [../]
  [./mat4]
    MatID = 4
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring4
    MGLibObject = RingLib4
  [../]
  [./mat5]
    MatID = 5
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring5
    MGLibObject = RingLib5
  [../]
  [./mat6]
    MatID = 6
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring6
    MGLibObject = RingLib6
  [../]
  [./mat7]
    MatID = 7
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring7
    MGLibObject = RingLib7
  [../]
  [./mat8]
    MatID = 8
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring8
    MGLibObject = RingLib8
  [../]
  [./mat9]
    MatID = 9
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring9
    MGLibObject = RingLib9
  [../]
  [./mat10]
    MatID = 10
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring10
    MGLibObject = RingLib10
  [../]
  [./mat11]
    MatID = 11
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring11
    MGLibObject = RingLib11
  [../]
  [./mat12]
    MatID = 12
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring12
    MGLibObject = RingLib12
  [../]
  [./mat13]
    MatID = 13
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring13
    MGLibObject = RingLib13
  [../]
  [./mat14]
    MatID = 14
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring14
    MGLibObject = RingLib14
  [../]
  [./mat15]
    MatID = 15
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring15
    MGLibObject = RingLib15
  [../]
  [./mat16]
    MatID = 16
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring16
    MGLibObject = RingLib16
  [../]
  [./mat17]
    MatID = 17
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring17
    MGLibObject = RingLib17
  [../]
  [./mat18]
    MatID = 18
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring18
    MGLibObject = RingLib18
  [../]
  [./mat19]
    MatID = 19
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring19
    MGLibObject = RingLib19
  [../]
  [./mat20]
    MatID = 20
    plus = true
    type = CoupledFeedbackNeutronicsMaterial
    block = ring20
    MGLibObject = RingLib20
  [../]
  [./mat21]
    MatID = 21
    type = CoupledFeedbackNeutronicsMaterial
    block = gap
    MGLibObject = GapLib
  [../]
  [./mat22]
    MatID = 22
    type = CoupledFeedbackNeutronicsMaterial
    block = clad
    MGLibObject = CladLib
  [../]
  [./mat23]
    MatID = 23
    type = CoupledFeedbackNeutronicsMaterial
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
  # used for controlling screen print-out
  #  picard_max_its = 1
  # postprocessors for eigen solve
  petsc_options_value = 'hypre boomeramg 0.7 100'
  type = Depletion
  solve_type = 'PJFNK'
  l_max_its = 200
  source_abs_tol = 1e-7
  petsc_options_iname = '-pc_type -pc_hypre_type -pc_hypre_boomeramg_strong_threshold -ksp_gmres_restart '
  burnup = ' 0.0 43200 86400 172800 864000 1728000 2592000 3456000 4320000 6480000 8640000 12960000 17280000 21600000 25920000 30240000'
  initial_free_power_iterations = 8
[]
[MultiApps]
  [./sub]
    app_type = BisonApp
    input_files = bison_2way_fuelcladelastic.i
    type = TransientMultiApp
    positions = '0.0 0.0 0.0'
  [../]
[]
[Transfers]
  [./tosub]
    variable = burnup
    source_variable = BISONBurnup
    direction = to_multiapp
    type = MultiAppInterpolationTransfer
    multi_app = sub
  [../]
  [./tosub2]
    variable = fission_rate
    source_variable = BISONFissionRate
    direction = to_multiapp
    type = MultiAppInterpolationTransfer
    multi_app = sub
  [../]
  [./tosub3]
    variable = power_density
    source_variable = BISONPowerDensity
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
  [./MammothTotalPower]
    variable = BISONPowerDensity
    type = ElementIntegralVariablePostprocessor
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10
ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
  [../]
[]
[Outputs]
  [./console]
    perf_log = true
    max_rows = 25
    type = Console
    linear_residuals = true
  [../]
  [./exodus]
    file_base =  QuarterPin_S2_2way
    type = Exodus
  [../]
[]
