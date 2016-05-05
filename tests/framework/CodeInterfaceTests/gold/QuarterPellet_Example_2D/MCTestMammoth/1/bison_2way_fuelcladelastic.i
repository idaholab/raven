[GlobalParams]
  #density = 10480.0
  energy_per_fission =  3.28451e-11
  family = LAGRANGE
  order = SECOND
  disp_y = disp_y
  disp_x = disp_x
[]
[Mesh]
  patch_size =  100
  displacements = 'disp_x disp_y'
  file = QuarterPin_FuelClad_FineQUAD8.e
[]
[Variables]
  [./disp_x]
    initial_condition = 0.0
  [../]
  [./disp_y]
    initial_condition = 0.0
  [../]
  [./temp]
    initial_condition =  602.0
  [../]
[]
[AuxVariables]
  [./fission_rate]
    family = MONOMIAL
    order = CONSTANT
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
  [../]
  [./burnup]
    family = MONOMIAL
    order = CONSTANT
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
  [../]
  [./power_density]
    family = MONOMIAL
    order = CONSTANT
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
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
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
    order = CONSTANT
    family = MONOMIAL
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
    block = clad
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./hoop_creep_strain]
    block = clad
    order = CONSTANT
    family = MONOMIAL
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
    y = '0 1'
    x = '0 3600'
    type = PiecewiseLinear
  [../]
[]
[SolidMechanics]
  [./solid]
    disp_y = disp_y
    temp = temp
    disp_x = disp_x
  [../]
[]
[Kernels]
  [./heat]
    # gradient term in heat conduction equation
    variable = temp
    type = HeatConduction
  [../]
  [./heat_ie]
    # time term in heat conduction equation
    variable = temp
    type = HeatConductionTimeDerivative
  [../]
  [./heat_source]
    # source term in heat conduction equation
    variable = temp
    energy_per_fission = 1
    type = NeutronHeatSource
    fission_rate = power_density
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
  [../]
[]
[AuxKernels]
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
  [./grain_radius]
    variable = grain_radius
    execute_on = residual
    type = GrainRadiusAux
    temp = temp
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
  [../]
  [./stress_xx]
    # computes stress components for output
    variable = stress_xx
    index = 0
    type = MaterialTensorAux
    tensor = stress
    execute_on =  timestep
  [../]
  [./stress_yy]
    variable = stress_yy
    index = 1
    type = MaterialTensorAux
    tensor = stress
    execute_on = timestep
  [../]
  [./stress_zz]
    variable = stress_zz
    index = 2
    type = MaterialTensorAux
    tensor = stress
    execute_on = timestep
  [../]
  [./vonmises]
    variable = vonmises
    execute_on = timestep
    type = MaterialTensorAux
    tensor = stress
    quantity = vonmises
  [../]
  [./hoop_stress]
    variable = hoop_stress
    execute_on = timestep
    type = MaterialTensorAux
    tensor = stress
    quantity = hoop
  [../]
  [./hydrostatic_stress]
    variable = hydrostatic_stress
    execute_on = residual
    type = MaterialTensorAux
    tensor = stress
    quantity = hydrostatic
  [../]
  [./vol_strain]
    variable = volumetric_strain
    execute_on = timestep
    type = MaterialTensorAux
    tensor = total_strain
    quantity = FirstInvariant
  [../]
  [./pid]
    variable = pid
    type = ProcessorIDAux
  [../]
  [./gap_cond]
    variable = gap_conductance
    boundary = 8
    property = gap_conductance
    type = MaterialRealAux
  [../]
[]
[Contact]
  # Define mechanical contact between the 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20' (sideset=10) and the clad (sideset=5)
  [./pellet_clad_mechanical]
    slave = 8
    disp_y = disp_y
    tangential_tolerance = 5e-4
    disp_x = disp_x
    penalty = 1e7
    master = 5
    model = experimental
    normal_smoothing_distance = 0.1
  [../]
[]
[ThermalContact]
  # Define thermal contact between the 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20' (sideset=10) and the clad (sideset=5)
  [./thermal_contact]
    #    tangential_tolerance = 1e-4
    #    quadrature = true
    #    min_gap = 10e-6  #default 1e-6
    variable = temp
    master = 7
    gap_conductivity = 10
    type = GapHeatTransferLWR
    slave = 8
  [../]
[]
[BCs]
  # Define boundary conditions
  [./no_y_all]
    # pin pellets and clad along axis of symmetry (y)
    variable = disp_y
    boundary = '1004'
    type = DirichletBC
    value = 0.0
  [../]
  [./no_x_all]
    # pin pellets and clad along axis of symmetry (x)
    variable = disp_x
    boundary = '1003'
    type = DirichletBC
    value = 0.0
  [../]
  [./Pressure]
    #  apply coolant pressure on clad outer walls
    [./coolantPressure]
        #BWR change: lower pressure
        function = coolant_pressure_ramp
        boundary = 2
        factor = 15e6
    [../]
  [../]
  [./TemperatureBC]
    variable = temp
    boundary = '2'
    type = DirichletBC
    value = 602
  [../]
[]
[Materials]
  # Define material behavior models and input material property data
  [./fuel_thermal]
    # temperature and burnup dependent thermal properties of UO2 (bison kernel)
    burnup = burnup
    model = 2
    type = ThermalFuel
    temp = temp
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
  [../]
  [./fuel_solid_mechanics_elastic]
    # general isotropic linear thermoelasticity input (elk kernel)
    disp_y = disp_y
    temp = temp
    disp_x = disp_x
    poissons_ratio = 0.307173369324
    thermal_expansion = 10.0e-6
    youngs_modulus = 2.e11
    type = Elastic
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
  [../]
  [./clad_solid_mechanics_elastic]
    # general isotropic linear thermoelasticity input (elk kernel)
    disp_y = disp_y
    temp = temp
    disp_x = disp_x
    poissons_ratio = 0.3
    thermal_expansion = 5.0e-6
    youngs_modulus = 7.5e10
    type = Elastic
    block = clad
  [../]
  [./clad_thermal]
    # general thermal property input (elk kernel)
    specific_heat = 330.0
    type = HeatConductionMaterial
    block = clad
    thermal_conductivity = 16.0
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
    variable = temp
    type = MaxIncrement
    max_increment = 100.0
  [../]
  [./limitX]
    variable = disp_x
    type = MaxIncrement
    max_increment = 1e-5
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
  #Preconditioned JFNK (default)
  #  end_time = 14400.00
  #  num_steps = 5000
  nl_abs_tol = 1e-6
  petsc_options_value = 'basic                    201                hypre    boomeramg      4'
  nl_max_its = 45
  type = Transient
  start_time = 0.0
  line_search = 'none'
  l_tol = 8e-3
  solve_type = 'PJFNK'
  l_max_its = 100
  dtmin = 1.0
  dt = 8.0e2
  petsc_options_iname = '-snes_linesearch_type -ksp_gmres_restart -pc_type -pc_hypre_type -pc_hypre_boomeramg_max_iter'
  end_time = 1000.00
[]
[Postprocessors]
  # Define postprocessors (some are required as specified above; others are optional; many others are available)
  [./clad_inner_vol]
    # volume inside of cladding
    boundary = 7
    type = InternalVolume
  [../]
  [./pellet_volume]
    # 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20' pellet total volume
    boundary = 8
    type = InternalVolume
  [../]
  [./gas_volume]
    boundary = 9
    type = InternalVolume
  [../]
  [./interior_temp]
    variable = temp
    boundary =  9
    type = SideAverageValue
  [../]
  [./flux_from_clad]
    variable = temp
    boundary = 5
    diffusivity = thermal_conductivity
    type = SideFluxIntegral
  [../]
  [./flux_from_fuel]
    variable = temp
    boundary = 8
    diffusivity = thermal_conductivity
    type = SideFluxIntegral
  [../]
  [./_dt]
    # time step
    type = TimestepSize
  [../]
  [./nonlinear_its]
    # number of nonlinear iterations at each timestep
    type = NumNonlinearIterations
  [../]
  [./rod_total_power]
    variable = temp
    type = ElementIntegralPower
    block = 'ring1 ring2 ring3 ring4 ring5 ring6 ring7 ring8 ring9 ring10 ring11 ring12 ring13 ring14 ring15 ring16 ring17 ring18 ring19 ring20'
    fission_rate = fission_rate
  [../]
[]
[Outputs]
  # Define output file(s)
  #perf_log = true
  output_initial = true
  interval = 1
  exodus = true
  [./exodus2]
    file_base =  BISON_S2_1Itr_Twoway
    type = Exodus
  [../]
  [./console]
    perf_log = true
    max_rows = 25
    type = Console
    linear_residuals = true
  [../]
[]
