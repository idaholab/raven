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
  grid_variables = 'burnup Temp'
  dbgmat = true
  ngroup = 8
  forDiffusion = false
  forAdjoint = false
  forTransient = false
  allowVacuum = true
[]

[TransportSystems]
  particle = neutron
  equation_type = eigenvalue
  G = 8
  ReflectingBoundary = '1001 1002 1003 1004'
  [./sn]
    scheme = SAAF-CFEM-SN
    family = LAGRANGE
    order = FIRST
    AQtype = Level-Symmetric
    AQorder = 2
  [../]
[]

[Depletion]
  transport_system = sn
  power_modulating_function = PowerModulator

  burnup_unit = MWd/kg
  # rated power in MW / m
  rated_power = 0.005
  # area of the fuel in m^2
  fuel_volume = 4138476.487714
  # density of the fuel in kg / m^3
  fuel_density = 10480.0
  # fuel composition
  isotopes = 'U235 U238 O16'
  weight_percentages = '3.922 84.222 11.856'
[]

[AuxVariables]
   [./Temp]
     initial_condition = 622.0
   [../]
   [./burnup_fima]
     family = MONOMIAL
     order = CONSTANT
     initial_condition = 0.0
   [../]
   [./fission_rate]
     family = MONOMIAL
     order = CONSTANT
     initial_condition = 0.0
   [../]
   [./total_reactor_power_density_watt]
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

 [./burnup_fima_aux]
# in unit of J/cm
   type = PowerDensityTimeIntegrator
   variable = burnup_fima
   variable_to_integrate = total_reactor_power_density
   burnup_unit_converter = burnup_converter_uo
   burnup_unit = FIMA
   block = 'ring1 ring2 ring3 ring4 ring5
            ring6 ring7 ring8 ring9 ring10
            ring11 ring12 ring13 ring14 ring15
            ring16 ring17 ring18 ring19 ring20
            gap clad water'
 [../]
 [./fission_rate_aux]
   type = VectorReactionRate
   scalar_flux = 'flux_moment_g0_L0_M0
                  flux_moment_g1_L0_M0
                  flux_moment_g2_L0_M0
                  flux_moment_g3_L0_M0
                  flux_moment_g4_L0_M0
                  flux_moment_g5_L0_M0
                  flux_moment_g6_L0_M0
                  flux_moment_g7_L0_M0'
   cross_section = sigma_fission
   block = 'ring1 ring2 ring3 ring4 ring5
             ring6 ring7 ring8 ring9 ring10
             ring11 ring12 ring13 ring14 ring15
             ring16 ring17 ring18 ring19 ring20'
   variable = fission_rate
   scale_factor = power_scaling
 [../]
 [./total_reactor_power_density_watt_aux]
   type = ScaleAux
   variable = total_reactor_power_density_watt
   source_variable = total_reactor_power_density
   multiplier = 1000
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
   material_id = 1
 [../]
 [./mat2]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring2
   MGLibObject = RingLib2
   plus = true
   material_id = 2
 [../]
 [./mat3]
    type = CoupledFeedbackNeutronicsMaterial
   block = ring3
   MGLibObject = RingLib3
   plus = true
   material_id = 3
 [../]
 [./mat4]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring4
   MGLibObject = RingLib4
   plus = true
   material_id = 4
 [../]
 [./mat5]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring5
   MGLibObject = RingLib5
   plus = true
   material_id = 5
 [../]
 [./mat6]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring6
   MGLibObject = RingLib6
   plus = true
   material_id = 6
 [../]
 [./mat7]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring7
   MGLibObject = RingLib7
   plus = true
   material_id = 7
 [../]
 [./mat8]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring8
   MGLibObject = RingLib8
   plus = true
   material_id = 8
 [../]
 [./mat9]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring9
   MGLibObject = RingLib9
   plus = true
   material_id = 9
 [../]
 [./mat10]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring10
   MGLibObject = RingLib10
   plus = true
   material_id = 10
 [../]
 [./mat11]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring11
   MGLibObject = RingLib11
   plus = true
   material_id = 11
 [../]
 [./mat12]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring12
   MGLibObject = RingLib12
   plus = true
   material_id = 12
 [../]
 [./mat13]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring13
   MGLibObject = RingLib13
   plus = true
   material_id = 13
 [../]
 [./mat14]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring14
   MGLibObject = RingLib14
   plus = true
   material_id = 14
 [../]
 [./mat15]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring15
   MGLibObject = RingLib15
   plus = true
   material_id = 15
 [../]
 [./mat16]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring16
   MGLibObject = RingLib16
   plus = true
   material_id = 16
 [../]
 [./mat17]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring17
   MGLibObject = RingLib17
   plus = true
   material_id = 17
 [../]
 [./mat18]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring18
   MGLibObject = RingLib18
   plus = true
   material_id = 18
 [../]
 [./mat19]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring19
   MGLibObject = RingLib19
   plus = true
   material_id = 19
 [../]
 [./mat20]
   type = CoupledFeedbackNeutronicsMaterial
   block = ring20
   MGLibObject = RingLib20
   plus = true
   material_id = 20
 [../]
 [./mat21]
   type = CoupledFeedbackNeutronicsMaterial
   block = gap
   MGLibObject = GapLib
   material_id = 21
 [../]
 [./mat22]
   type = CoupledFeedbackNeutronicsMaterial
   block = clad
   MGLibObject = CladLib
   material_id = 22
 [../]
 [./mat23]
   type = CoupledFeedbackNeutronicsMaterial
   block = water
   MGLibObject = WaterLib
   material_id = 23
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
  initial_free_power_iterations = 4
# number of free power iterations for each individual eigen solve
# absolute convergence on residual for each individual eigen solve
  source_abs_tol = 1e-7
  picard_max_its = 10
[]

[MultiApps]
  [./sub]
    type = TransientMultiApp
    app_type = BisonApp
    positions = '0.0 0.0 0.0'
    input_files = bison_2way_fuel_clad_elastic.i
  [../]
[]

[Transfers]
  [./tosub]
    type = MultiAppInterpolationTransfer
    direction = to_multiapp
    multi_app = sub
    source_variable = burnup_fima
    variable = burnup
  [../]
  [./tosub2]
    type = MultiAppInterpolationTransfer
    direction = to_multiapp
    multi_app = sub
    source_variable = fission_rate
    variable = fission_rate
  [../]
  [./tosub3]
    type = MultiAppInterpolationTransfer
    direction = to_multiapp
    multi_app = sub
    source_variable = total_reactor_power_density_watt
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
  [./FuelArea]
    type = VolumePostprocessor
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10
             ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
  [../]
  [./burnup_MWdkg_pp]
    type = ElementAverageValue
    variable = burnup
  [../]
  [./burnup_FIMA_pp]
    type = ElementAverageValue
    variable = burnup_fima
  [../]
  [./TotalReactorPower]
     type = ElementIntegralVariablePostprocessor
     variable = total_reactor_power_density
  [../]
[]


[Outputs]
  interval = 1
  csv = true
  [./console]
    type = Console
    perf_log = true
    linear_residuals = true
    max_rows = 25
  [../]
  [./exodus]
    type = Exodus
    file_base = quarter_pin_with_fuel
  [../]
[]
