[Mesh]
  file = QuarterPin_FuelGapCladWater_QUAD8_Fine.e
  uniform_refine = 0
  second_order = true
  distribution = serial
[]

[GlobalParams]
  plus = true
  isMeter = true
  isotopes = 'pseudo'
  densities = 1.0
  grid_names = 'Burnup Tfuel'
  grid_variables = 'fuelburnup Temp'
  burnup_grid = 1
  dbgmat = true
  ngroup = 8
  forDiffusion = false
  forAdjoint = false
  forTransient = false
 allowVacuum = true
[]

[RattleSnakeParams]
  calculation_type = SAAF
  AQtype = Level-Symmetric
  AQorder = 2

  order = SECOND

  p0aux2mat = true
  forceflux = true
  verbose = 3
  hide_angular_flux = true
  hide_higher_flux_moment =0

  G = 8
  NA = 2  #Number of anisotropy
  n_delay_groups = 0

  eigenvalue = true
  transient = 0
  depletion = true               #Depletion boolean

  ReflectingBoundary = '1001 1002 1003 1004'
[]

[MacroDepletion]
  calculation_type = SAAF
  G = 8
  add_fission_rate = true
  family = MONOMIAL
  order = CONSTANT
  block = 'ring1 ring2 ring3 ring4 ring5
           ring6 ring7 ring8 ring9 ring10
           ring11 ring12 ring13 ring14 ring15
           ring16 ring17 ring18 ring19 ring20'
## in unit of W/cm for the pin
  rated_power = 4800.0
### pin volume
  burnup_volume = 1.0
  verbose = 2
  power_modulating_function = PowerModulator
[]

[AuxVariables]
   [./Temp]
     initial_condition = 622.0
   [../]
   [./fuelburnup]
     family = MONOMIAL
     order = CONSTANT
     initial_condition = 0.0
   [../]
   [./BISONBurnup]
     family = MONOMIAL
     order = CONSTANT
     initial_condition = 0.0
   [../]
   [./BISONFissionRate]
     family = MONOMIAL
     order = CONSTANT
     initial_condition = 0.0
   [../]
   [./BISONPowerDensity]
     family = MONOMIAL
     order = CONSTANT
     initial_condition = 0.0
   [../]
[]


[AuxKernels]
#
# Setting the fuel temperature
#
 [./SetTCool]
   block = water
   type = ConstantAux
   variable = Temp
   value = 622.0
 [../]
 [./CalcBurnup]
# in unit of J/cm
   type = PowerDensityTimeIntegrator
   variable = fuelburnup
   variable_to_integrate = power_density
   burnup_unit_converter = BurnupConverter
   burnup_unit = MWdkg
   block = 'ring1 ring2 ring3 ring4 ring5
            ring6 ring7 ring8 ring9 ring10
            ring11 ring12 ring13 ring14 ring15
            ring16 ring17 ring18 ring19 ring20
            gap clad water'
   time_coef = 0.000011574
 [../]
 [./CalcBISONBurnup]
# in unit of J/cm
   type = PowerDensityTimeIntegrator
   variable = BISONBurnup
   variable_to_integrate = power_density
   burnup_unit_converter = BurnupConverter
   burnup_unit = FIMA
   block = 'ring1 ring2 ring3 ring4 ring5
            ring6 ring7 ring8 ring9 ring10
            ring11 ring12 ring13 ring14 ring15
            ring16 ring17 ring18 ring19 ring20
            gap clad water'
   time_coef = 0.000011574
 [../]
 [./CalcBISONPowerDensity]
   type = ScaledVectorReactionRate
   scalar_flux = 'flux_moment_g0_L0_M0
                   flux_moment_g1_L0_M0
                   flux_moment_g2_L0_M0
                   flux_moment_g3_L0_M0
                   flux_moment_g4_L0_M0
                   flux_moment_g5_L0_M0
                   flux_moment_g6_L0_M0
                   flux_moment_g7_L0_M0'
   sigma_xs = kappa_sigma_fission
   block = 'ring1 ring2 ring3 ring4 ring5
             ring6 ring7 ring8 ring9 ring10
             ring11 ring12 ring13 ring14 ring15
             ring16 ring17 ring18 ring19 ring20'
   variable = BISONPowerDensity
   pp_scale_factor = power_scaling
 [../]
 [./CalcBISONFissionRate]
   type = ScaledVectorReactionRate
   scalar_flux = 'flux_moment_g0_L0_M0
                   flux_moment_g1_L0_M0
                   flux_moment_g2_L0_M0
                   flux_moment_g3_L0_M0
                   flux_moment_g4_L0_M0
                   flux_moment_g5_L0_M0
                   flux_moment_g6_L0_M0
                   flux_moment_g7_L0_M0'
   sigma_xs = kappa_sigma_fission
   block = 'ring1 ring2 ring3 ring4 ring5
             ring6 ring7 ring8 ring9 ring10
             ring11 ring12 ring13 ring14 ring15
             ring16 ring17 ring18 ring19 ring20'
   variable = BISONFissionRate
   pp_scale_factor = power_scaling
   scale_factor = 3.0445941708e+10
 [../]
[]

[UserObjects]
   [./BurnupConverter]
     type = BurnupConverterObject
     power_density = 420.248
     density_UO2 = 10480.0
     heavy_metal_isotopes = 'U235 U238'
     weight_percentages = '4.45 95.55'
   [../]
[]

[YAKXSLibraries]
   [./RingLib1]
     type = BaseLibObject
     library_file = M1_2D.xml
     library_name = ring1
     library_type = MultigroupLibrary
     debug = 3
   [../]
   [./RingLib2]
     type = BaseLibObject
     library_file = M2_2D.xml
     library_name = ring2
     library_type = MultigroupLibrary
   [../]
   [./RingLib3]
     type = BaseLibObject
     library_file = M3_2D.xml
     library_name = ring3
     library_type = MultigroupLibrary
   [../]
   [./RingLib4]
     type = BaseLibObject
     library_file = M4_2D.xml
     library_name = ring4
     library_type = MultigroupLibrary
   [../]
   [./RingLib5]
     type = BaseLibObject
     library_file = M5_2D.xml
     library_name = ring5
     library_type = MultigroupLibrary
   [../]
   [./RingLib6]
     type = BaseLibObject
     library_file = M6_2D.xml
     library_name = ring6
     library_type = MultigroupLibrary
   [../]
   [./RingLib7]
     type = BaseLibObject
     library_file = M7_2D.xml
     library_name = ring7
     library_type = MultigroupLibrary
   [../]
   [./RingLib8]
     type = BaseLibObject
     library_file = M8_2D.xml
     library_name = ring8
     library_type = MultigroupLibrary
   [../]
   [./RingLib9]
     type = BaseLibObject
     library_file = M9_2D.xml
     library_name = ring9
     library_type = MultigroupLibrary
   [../]
   [./RingLib10]
     type = BaseLibObject
     library_file = M10_2D.xml
     library_name = ring10
     library_type = MultigroupLibrary
   [../]
   [./RingLib11]
     type = BaseLibObject
     library_file = M11_2D.xml
     library_name = ring11
     library_type = MultigroupLibrary
   [../]
   [./RingLib12]
     type = BaseLibObject
     library_file = M12_2D.xml
     library_name = ring12
     library_type = MultigroupLibrary
   [../]
   [./RingLib13]
     type = BaseLibObject
     library_file = M13_2D.xml
     library_name = ring13
     library_type = MultigroupLibrary
   [../]
   [./RingLib14]
     type = BaseLibObject
     library_file = M14_2D.xml
     library_name = ring14
     library_type = MultigroupLibrary
   [../]
   [./RingLib15]
     type = BaseLibObject
     library_file = M15_2D.xml
     library_name = ring15
     library_type = MultigroupLibrary
   [../]
   [./RingLib16]
     type = BaseLibObject
     library_file = M16_2D.xml
     library_name = ring16
     library_type = MultigroupLibrary
   [../]
   [./RingLib17]
     type = BaseLibObject
     library_file = M17_2D.xml
     library_name = ring17
     library_type = MultigroupLibrary
   [../]
   [./RingLib18]
     type = BaseLibObject
     library_file = M18_2D.xml
     library_name = ring18
     library_type = MultigroupLibrary
   [../]
   [./RingLib19]
     type = BaseLibObject
     library_file = M19_2D.xml
     library_name = ring19
     library_type = MultigroupLibrary
   [../]
   [./RingLib20]
     type = BaseLibObject
     library_file = M20_2D.xml
     library_name = ring20
     library_type = MultigroupLibrary
   [../]
   [./GapLib]
     type = BaseLibObject
     library_file = M21_2D.xml
     library_name = gap
     library_type = MultigroupLibrary
   [../]
   [./CladLib]
     type = BaseLibObject
     library_file = M22_2D.xml
     library_name = clad
     library_type = MultigroupLibrary
   [../]
   [./WaterLib]
     type = BaseLibObject
     library_file = M23_2D.xml
     library_name = water
     library_type = MultigroupLibrary
   [../]
[]


[Materials]
 [./mat1]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring1
   MGLibObject = RingLib1
   plus = true
   MatID = 1
 [../]
 [./mat2]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring2
   MGLibObject = RingLib2
   plus = true
   MatID = 2
 [../]
 [./mat3]
    type = CoupledFeedbackNeutronicsMaterial
   block = ring3
   MGLibObject = RingLib3
   plus = true
   MatID = 3
 [../]
 [./mat4]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring4
   MGLibObject = RingLib4
   plus = true
   MatID = 4
 [../]
 [./mat5]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring5
   MGLibObject = RingLib5
   plus = true
   MatID = 5
 [../]
 [./mat6]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring6
   MGLibObject = RingLib6
   plus = true
   MatID = 6
 [../]
 [./mat7]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring7
   MGLibObject = RingLib7
   plus = true
   MatID = 7
 [../]
 [./mat8]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring8
   MGLibObject = RingLib8
   plus = true
   MatID = 8
 [../]
 [./mat9]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring9
   MGLibObject = RingLib9
   plus = true
   MatID = 9
 [../]
 [./mat10]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring10
   MGLibObject = RingLib10
   plus = true
   MatID = 10
 [../]
 [./mat11]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring11
   MGLibObject = RingLib11
   plus = true
   MatID = 11
 [../]
 [./mat12]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring12
   MGLibObject = RingLib12
   plus = true
   MatID = 12
 [../]
 [./mat13]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring13
   MGLibObject = RingLib13
   plus = true
   MatID = 13
 [../]
 [./mat14]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring14
   MGLibObject = RingLib14
   plus = true
   MatID = 14
 [../]
 [./mat15]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring15
   MGLibObject = RingLib15
   plus = true
   MatID = 15
 [../]
 [./mat16]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring16
   MGLibObject = RingLib16
   plus = true
   MatID = 16
 [../]
 [./mat17]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring17
   MGLibObject = RingLib17
   plus = true
   MatID = 17
 [../]
 [./mat18]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring18
   MGLibObject = RingLib18
   plus = true
   MatID = 18
 [../]
 [./mat19]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring19
   MGLibObject = RingLib19
   plus = true
   MatID = 19
 [../]
 [./mat20]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring20
   MGLibObject = RingLib20
   plus = true
   MatID = 20
 [../]
 [./mat21]
   type = CoupledFeedbackNeutronicsMaterial
   block = gap
   MGLibObject = GapLib
   MatID = 21
 [../]
 [./mat22]
   type = CoupledFeedbackNeutronicsMaterial
   block = clad
   MGLibObject = CladLib
   MatID = 22
 [../]
 [./mat23]
   type = CoupledFeedbackNeutronicsMaterial
   block = water
   MGLibObject = WaterLib
   MatID = 23
 [../]
[]

[Functions]
  [./PowerModulator]
    type = PiecewiseLinear
    x = '0 43200.0 30240000'
    y = '0 1.0 1.0'
  [../]
[]

[Executioner]
  type = Depletion
# general PetSc settings
  solve_type = 'PJFNK'
  petsc_options_iname = '-pc_type -pc_hypre_type -pc_hypre_boomeramg_strong_threshold -ksp_gmres_restart '
  petsc_options_value = 'hypre boomeramg 0.7 100'
  l_max_its = 200

# in unit of days
  burnup = ' 0.0 43200 86400 172800 864000 1728000 2592000 3456000 4320000 6480000 8640000 12960000 17280000 21600000 25920000 30240000'
# extra number of free power iterations for solving time 0
  initial_free_power_iterations = 8
# number of free power iterations for each individual eigen solve
# absolute convergence on residual for each individual eigen solve
  source_abs_tol = 1e-7
# used for controlling screen print-out
#  picard_max_its = 1

# postprocessors for eigen solve
[]

[MultiApps]
  [./sub]
    type = TransientMultiApp
    app_type = BisonApp
    positions = '0.0 0.0 0.0'
    input_files = bison_2way_fuelcladelastic.i
  [../]
[]

[Transfers]
  [./tosub]
    type = MultiAppInterpolationTransfer
    direction = to_multiapp
    multi_app = sub
    source_variable = BISONBurnup
    variable = burnup
  [../]
  [./tosub2]
    type = MultiAppInterpolationTransfer
    direction = to_multiapp
    multi_app = sub
    source_variable = BISONFissionRate
    variable = fission_rate
  [../]
  [./tosub3]
    type = MultiAppInterpolationTransfer
    direction = to_multiapp
    multi_app = sub
    source_variable = BISONPowerDensity
    variable = power_density
   [../]
   [./fromsub]
    type = MultiAppInterpolationTransfer
    direction = from_multiapp
    multi_app = sub
    source_variable = temp
    variable = Temp
   [../]
[]

[Postprocessors]
  [./MammothTotalPower]
     type = ElementIntegralVariablePostprocessor
     variable = BISONPowerDensity
     block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10
              ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
  [../]
[]


[Outputs]
 [./console]
    type = Console
    perf_log = true
    linear_residuals = true
    max_rows = 25
 [../]
 [./exodus]
 type = Exodus
 file_base = QuarterPin_S2_2way  # set the file base (the extension is automatically applied)
  [../]

[]
