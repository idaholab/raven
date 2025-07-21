[GlobalParams]
  disp_x = disp_x
  disp_y = disp_y
  order = SECOND
  family = LAGRANGE
  energy_per_fission = 3.28451e-11  # J/fission
  density = 10480.0
[]

[Mesh]
  file = QuarterPin_FuelClad_FineQUAD8.e
  displacements = 'disp_x disp_y'
  patch_size = 100 # For contact algorithm
[]

[Variables]
  [./disp_x]
    initial_condition = 0.0
  [../]

  [./disp_y]
    initial_condition = 0.0
  [../]

  [./temp]
    initial_condition = 602.0     # set initial temp to hot zero power
  [../]
[]

[AuxVariables]
  [./fission_rate]
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
    order = CONSTANT
    family = MONOMIAL
  [../]

  [./burnup]
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
    order = CONSTANT
    family = MONOMIAL
  [../]

  [./power_density]
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
    order = CONSTANT
    family = MONOMIAL
  [../]

  [./fast_neutron_flux]
    block = clad
  [../]

  [./fast_neutron_fluence]
    block = clad
  [../]

  [./grain_radius]
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
    initial_condition = 5e-6
  [../]
  [./volumetric_strain]
    order = CONSTANT
    family = MONOMIAL
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
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
  [./relocation_strain]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./vonmises]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./hoop_stress]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./hydrostatic_stress]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./eff_creep_strain]
    order = CONSTANT
    family = MONOMIAL
    block = clad
  [../]
  [./hoop_creep_strain]
    order = CONSTANT
    family = MONOMIAL
    block = clad
  [../]
  [./gap_conductance]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./pid]
    order = CONSTANT
    family = MONOMIAL
  [../]
[]

[Functions]
  [./coolant_pressure_ramp]
    type = PiecewiseLinear
    x = '0 3600'
    y = '0 1'
  [../]

[]

[SolidMechanics]
  [./solid]
    temp = temp
    disp_x = disp_x
    disp_y = disp_y
  [../]
[]

[Kernels]
  [./heat]         # gradient term in heat conduction equation
    type = HeatConduction
    variable = temp
  [../]

  [./heat_ie]       # time term in heat conduction equation
    type = HeatConductionTimeDerivative
    variable = temp
  [../]

  [./heat_source]  # source term in heat conduction equation
     type = NeutronHeatSource
     variable = temp
     block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
     fission_rate = power_density
     energy_per_fission = 1
  [../]
[]

[AuxKernels]
  [./grain_radius]
    type = GrainRadiusAux
     block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
    variable = grain_radius
    temp = temp
    execute_on = linear
  [../]
  [./stress_xx]               # computes stress components for output
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

  [./hoop_stress]
    type = MaterialTensorAux
    tensor = stress
    variable = hoop_stress
    quantity = hoop
    execute_on = timestep_end
  [../]
  [./hydrostatic_stress]
    type = MaterialTensorAux
    tensor = stress
    variable = hydrostatic_stress
    quantity = hydrostatic
    execute_on = linear
  [../]
#  [./eff_creep_strain]
#    type = MaterialTensorAux
#    tensor = creep_strain
#    variable = eff_creep_strain
#    quantity = plasticstrainmag
#    execute_on = timestep
#  [../]
#  [./hoop_creep_strain]
#    type = MaterialTensorAux
#    tensor = creep_strain
#    variable = hoop_creep_strain
#    quantity = hoop
#    execute_on = timestep
#  [../]
  [./vol_strain]
    type = MaterialTensorAux
    tensor = total_strain
    variable = volumetric_strain
    quantity = FirstInvariant
    execute_on = timestep_end
  [../]

  [./pid]
    type = ProcessorIDAux
    variable = pid
  [../]
  [./gap_cond]
    type = MaterialRealAux
    property = gap_conductance
    variable = gap_conductance
    boundary = 8
  [../]

[]

[Contact]
  # Define mechanical contact between the 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20' (sideset=10) and the clad (sideset=5)
  [./pellet_clad_mechanical]
    master = 5
    slave = 8
    disp_x = disp_x
    disp_y = disp_y
    penalty = 1e7
    tangential_tolerance = 5e-4
    model = experimental
    normal_smoothing_distance = 0.1
  [../]
[]

[ThermalContact]
  # Define thermal contact between the 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20' (sideset=10) and the clad (sideset=5)
  [./thermal_contact]
    type = GapHeatTransferLWR
    variable = temp
    master = 7
    slave = 8
    gap_conductivity = 10
#    tangential_tolerance = 1e-4
#    quadrature = true
#    min_gap = 10e-6  #default 1e-6
  [../]
[]



[BCs]
# Define boundary conditions

  [./no_y_all] # pin pellets and clad along axis of symmetry (y)
    type = DirichletBC
    variable = disp_y
    boundary = '1004'
    value = 0.0
  [../]

  [./no_x_all] # pin pellets and clad along axis of symmetry (x)
    type = DirichletBC
    variable = disp_x
    boundary = '1003'
    value = 0.0
  [../]

  [./Pressure] #  apply coolant pressure on clad outer walls
    [./coolantPressure]
      boundary = 2
      #BWR change: lower pressure
      factor = 15e6
      function = coolant_pressure_ramp
    [../]
  [../]

   [./TemperatureBC]
     type = DirichletBC
     value = 602
     variable = temp
     boundary = '2'
   [../]

[]

[Materials]
  # Define material behavior models and input material property data

  [./fuel_thermal]                       # temperature and burnup dependent thermal properties of UO2 (bison kernel)
    type = ThermalFuel
    model = 2
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
    temp = temp
    burnup = burnup
  [../]

  [./fuel_solid_mechanics_elastic]        # general isotropic linear thermoelasticity input (elk kernel)
     type = Elastic
     block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
     disp_x = disp_x
     disp_y = disp_y
     temp = temp
     youngs_modulus = 2.e11
     poissons_ratio = 0.345
     thermal_expansion = 10.0e-6
   [../]

  [./clad_solid_mechanics_elastic]        # general isotropic linear thermoelasticity input (elk kernel)
     type = Elastic
     block = clad
     disp_x = disp_x
     disp_y = disp_y
     temp = temp
     youngs_modulus = 7.5e10
     poissons_ratio = 0.3
     thermal_expansion = 5.0e-6
  [../]

  [./clad_thermal]                       # general thermal property input (elk kernel)
    type = HeatConductionMaterial
    block = clad
    thermal_conductivity = 16.0
    specific_heat = 330.0
  [../]

  [./clad_density]
    type = Density
    block = clad
    density = 6551.0
  [../]
  [./fuel_density]
    type = Density
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
    density = 10480.0
  [../]
[]


[Dampers]
  [./limitT]
    type = MaxIncrement
    max_increment = 100.0
    variable = temp
  [../]
  [./limitX]
    type = MaxIncrement
    max_increment = 1e-5
    variable = disp_x
  [../]
[]


[Executioner]

  # PETSC options:
  #   petsc_options
  #   petsc_options_iname
  #   petsc_options_value
  #
  # controls for linear iterations
  #   l_max_its
  #   l_tol
  #
  # controls for nonlinear iterations
  #   nl_max_its
  #   nl_rel_tol
  #   nl_abs_tol
  #
  # time control
  #   start_time
  #   dt
  #   optimal_iterations
  #   iteration_window
  #   linear_iteration_ratio

  type = Transient

  #Preconditioned JFNK (default)
  solve_type = 'PJFNK'

  petsc_options_iname = '-snes_linesearch_type -ksp_gmres_restart -pc_type -pc_hypre_type -pc_hypre_boomeramg_max_iter -pc_hypre_boomeramg_tol'
  petsc_options_value = 'basic                    201                hypre    boomeramg      10 1e-4'
  line_search = 'none'

  l_max_its = 100
  l_tol = 1.0e-5

  nl_max_its = 100
  nl_abs_tol = 1e-8
  nl_rel_tol = 1e-8

  start_time = 0.0
#  dt = 8.0e2
#  dtmin = 1.0
#  end_time = 14400.00
#  num_steps = 5000

[]

[Postprocessors]
  # Define postprocessors (some are required as specified above; others are optional; many others are available)

  [./clad_inner_vol]              # volume inside of cladding
   type = InternalVolume
    boundary = 7
  [../]

  [./pellet_volume]  # 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20' pellet total volume
    type = InternalVolume
    boundary = 8
  [../]

   [./gas_volume]
    type = InternalVolume
    boundary = 9
  [../]

  [./interior_temp]
    type = SideAverageValue
    boundary = 9 #bws was 9
    variable = temp
  [../]

  [./flux_from_clad]
    type = SideFluxIntegral
    variable = temp
    boundary = 5
    diffusivity = thermal_conductivity
  [../]

  [./flux_from_fuel]
    type = SideFluxIntegral
    variable = temp
    boundary = 8
    diffusivity = thermal_conductivity
  [../]

  [./_dt]                     # time step
    type =  TimestepSize
  [../]

  [./nonlinear_its]           # number of nonlinear iterations at each timestep
    type = NumNonlinearIterations
  [../]

  [./rod_total_power]
    type = ElementIntegralPower
    variable = temp
    fission_rate = fission_rate
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
  [../]

[]

[Outputs]
  # Define output file(s)

  interval = 1
  output_initial = true
  exodus = true
  perf_log = true
 [./exodus2]
 type = Exodus
   file_base = BISON_S2_1Itr_Twoway  # set the file base (the extension is automatically applied)
 [../]
  [./console]
    type = Console
    perf_log = true
    linear_residuals = true
    max_rows = 25
  [../]
[]
