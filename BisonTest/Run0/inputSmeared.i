
[GlobalParams]
  # Set initial fuel density, other global parameters
  density = 10431.0
  disp_x = disp_x
  disp_y = disp_y
  order = SECOND
  family = LAGRANGE
  energy_per_fission = 3.2e-11  # J/fission
[]

[Problem]
  # Specify coordinate system type
  coord_type = RZ
[]

[Mesh]
  # Import mesh file
  file = smeared.e
  displacements = 'disp_x disp_y'
  patch_size = 1000 # For contact algorithm
[]

[Variables]
  # Define dependent variables and initial conditions

  [./disp_x]
  [../]

  [./disp_y]
  [../]

  [./temp]
    initial_condition = 580.0     # set initial temp to coolant inlet
  [../]
[]

[AuxVariables]
  # Define auxilary variables

  [./fast_neutron_flux]
    block = clad
  [../]

  [./fast_neutron_fluence]
    block = clad
  [../]
  [./grain_radius]
    block = pellet_type_2
    initial_condition = 10e-6
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
  [./vonmises]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./creep_strain_mag]
    order = CONSTANT
    family = MONOMIAL
  [../]

  [./gap_cond]
    order = CONSTANT
    family = MONOMIAL
  [../]

  [./coolant_htc]
    order = CONSTANT
    family = MONOMIAL
  [../]
[]

[Functions]
  # Define functions to control power and boundary conditions

  [./power_history]
    type = PiecewiseLinearFile   # reads and interpolates an input file containing rod average linear power vs time
    yourFileName = powerhistory.csv
    scale_factor = 1
  [../]

  [./axial_peaking_factors]      # reads and interpolates an input file containing the axial power profile vs time
    type = PiecewiseBilinear
    yourFileName = peakingfactors12.csv
    scale_factor = 1
    axis = 1 # (0,1,2) => (x,y,z)
  [../]

  [./pressure_ramp]              # reads and interpolates input data defining amplitude curve for fill gas pressure
    type = PiecewiseLinear
    x = '-200 0'
    y = '0 1'
  [../]

  [./q]
    type = CompositeFunction
    functions = 'power_history axial_peaking_factors'
  [../]
[]

[SolidMechanics]
  # Specify that we need solid mechanics (divergence of stress)
  [./solid]
    disp_r = disp_x
    disp_z = disp_y
    temp = temp
  [../]
[]

[Kernels]
  # Define kernels for the various terms in the PDE system

  [./gravity]       # body force term in stress equilibrium equation
    type = Gravity
    variable = disp_y
    value = -9.81
  [../]

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
     block = pellet_type_2     # fission rate applied to the fuel (block 2) only
     fission_rate = fission_rate  # coupling to the fission_rate aux variable
  [../]
[]

[Burnup]
  [./burnup]
    block = pellet_type_2
    rod_ave_lin_pow = power_history          # using the power function defined above
    axial_power_profile = axial_peaking_factors     # using the axial power profile function defined above
    num_radial = 80
    num_axial = 11
    a_lower = 0.00944   # mesh dependent
    a_upper = 0.12804   # mesh dependent
    fuel_inner_radius = 0
    fuel_outer_radius = .0041
    fuel_volume_ratio = 1.0 # for use with dished pellets (ratio of actual volume to cylinder volume)

    #N235 = N235 # Activate to write N235 concentration to output file
    #N238 = N238 # Activate to write N238 concentration to output file
    #N239 = N239 # Activate to write N239 concentration to output file
    #N240 = N240 # Activate to write N240 concentration to output file
    #N241 = N241 # Activate to write N241 concentration to output file
    #N242 = N242 # Activate to write N242 concentration to output file
    RPF = RPF
  [../]
[]

[AuxKernels]
  # Define auxilliary kernels for each of the aux variables

  [./fast_neutron_flux]
    type = FastNeutronFluxAux
    variable = fast_neutron_flux
    block = clad
    rod_ave_lin_pow = power_history
    axial_power_profile = axial_peaking_factors
    factor = 3e13
    execute_on = timestep_begin
  [../]

  [./fast_neutron_fluence]
    type = FastNeutronFluenceAux
    variable = fast_neutron_fluence
    block = clad
    fast_neutron_flux = fast_neutron_flux
    execute_on = timestep_begin
  [../]

  [./grain_radius]
    type = GrainRadiusAux
    block = pellet_type_2
    variable = grain_radius
    temp = temp
    execute_on = residual
  [../]

  [./stress_xx]               # computes stress components for output
    type = MaterialTensorAux
    tensor = stress
    variable = stress_xx
    index = 0
    execute_on = timestep     # for efficiency, only compute at the end of a timestep
  [../]
  [./stress_yy]
    type = MaterialTensorAux
    tensor = stress
    variable = stress_yy
    index = 1
    execute_on = timestep
  [../]
  [./stress_zz]
    type = MaterialTensorAux
    tensor = stress
    variable = stress_zz
    index = 2
    execute_on = timestep
  [../]
  [./vonmises]
    type = MaterialTensorAux
    tensor = stress
    variable = vonmises
    quantity = vonmises
    execute_on = timestep
  [../]
  [./creep_strain_mag]
    type = MaterialTensorAux
    block = 'clad pellet_type_2'
    tensor = creep_strain
    variable = creep_strain_mag
    quantity = plasticstrainmag
    execute_on = timestep
  [../]

[]

[AuxBCs]
  [./conductance]
    type = MaterialRealAux
    property = gap_conductance
    variable = gap_cond
    boundary = 10
  [../]
  [./coolant_htc]
    type = MaterialRealAux
    property = coolant_channel_htc
    variable = coolant_htc
    boundary = '1 2 3'
  [../]
[]

[Contact]
  # Define mechanical contact between the fuel (sideset=10) and the clad (sideset=5)
  [./pellet_clad_mechanical]
    master = 5
    slave = 10
    disp_x = disp_x
    disp_y = disp_y
    penalty = 1e7
  [../]
[]

[ThermalContact]
  # Define thermal contact between the fuel (sideset=10) and the clad (sideset=5)
  [./thermal_contact]
    type = GapHeatTransferLWR
    variable = temp
    master = 7
    slave = 8
    initial_moles = initial_moles       # coupling to a postprocessor which supplies the initial plenum/gap gas mass
    gas_released = fis_gas_released     # coupling to a postprocessor which supplies the fission gas addition
    quadrature = true
    contact_pressure = contact_pressure
  [../]
  [./thermal_contact_bottom]
    type = GapHeatTransferLWR
    variable = temp
    master = 22
    slave = 21
    initial_moles = initial_moles       # coupling to a postprocessor which supplies the initial plenum/gap gas mass
    gas_released = fis_gas_released     # coupling to a postprocessor which supplies the fission gas addition
    quadrature = true
    contact_pressure = contact_pressure
  [../]
  [./thermal_contact_top]
    type = GapHeatTransferLWR
    variable = temp
    master = 23
    slave = 24
    initial_moles = initial_moles       # coupling to a postprocessor which supplies the initial plenum/gap gas mass
    gas_released = fis_gas_released     # coupling to a postprocessor which supplies the fission gas addition
    quadrature = true
    contact_pressure = contact_pressure
  [../]
[]

[BCs]
# Define boundary conditions

  [./no_x_all] # pin pellets and clad along axis of symmetry (y)
    type = DirichletBC
    variable = disp_x
    boundary = 12
    value = 0.0
  [../]

  [./no_y_clad_bottom] # pin clad bottom in the axial direction (y)
    type = DirichletBC
    variable = disp_y
    boundary = '1'
    value = 0.0
  [../]

  [./no_y_fuel_bottom] # pin fuel bottom in the axial direction (y)
    type = DirichletBC
    variable = disp_y
    boundary = '1020'
    value = 0.0
  [../]


  [./Pressure] #  apply coolant pressure on clad outer walls
    [./coolantPressure]
      boundary = '1 2 3'
      factor = 15.5e6
      function = pressure_ramp   # use the pressure_ramp function defined above
    [../]
  [../]

  [./PlenumPressure] #  apply plenum pressure on clad inner walls and pellet surfaces
    [./plenumPressure]
      boundary = 9
      initial_pressure = 2.0e6
      startup_time = -200
      R = 8.3143
      output_initial_moles = initial_moles       # coupling to post processor to get initial fill gas mass
      temperature = plenumTemp                   # coupling to post processor to get gas temperature approximation
      volume = gas_volume                        # coupling to post processor to get gas volume
      material_input = fis_gas_released          # coupling to post processor to get fission gas added
      output = plenum_pressure                   # coupling to post processor to output plenum/gap pressure
    [../]
  [../]

[]

[CoolantChannel]
  [./convective_clad_surface] # apply convective boundary to clad outer surface
    boundary = '1 2 3'
    variable = temp
    inlet_temperature = 580      # K
    inlet_pressure    = 15.5e6   # Pa
    inlet_massflux    = 3800     # kg/m^2-sec
    rod_diameter      = 0.948e-2 # m
    rod_pitch         = 1.26e-2  # m
    linear_heat_rate  = power_history
    axial_power_profile = axial_peaking_factors
  [../]
[]

[PlenumTemperature]
  [./plenumTemp]
    num_pellets = 3
    temp = temp
  [../]
[]

[Materials]
  # Define material behavior models and input material property data

  [./hafnium_thermal]
    type = HeatConductionMaterial
    block = 'pellet_type_1 pellet_type_3'
    thermal_conductivity = 2.0
    specific_heat = 286.5
  [../]
  [./hafnium_density]
    type = Density
    block = 'pellet_type_1 pellet_type_3'
    density = 9680.0
    disp_r = disp_x
    disp_z = disp_y
  [../]
  [./hafnium_elastic]
    type = Elastic
    block = 'pellet_type_1 pellet_type_3'
    youngs_modulus = 130e9
    poissons_ratio = 0.26
    temp = temp
    thermal_expansion = 1.4732e-07
    disp_r = disp_x
    disp_z = disp_y
    stress_free_temperature = 295.0
  [../]

  [./fuel_thermal]                       # temperature and burnup dependent thermal properties of UO2 (bison kernel)
    type = ThermalUO2
    block = pellet_type_2
    temp = temp
    burnup = burnup
  [../]

  [./fuel_solid_mechanics_swelling]      # free expansion strains (swelling and densification) for UO2 (bison kernel)
    type = VSwellingUO2
    block = pellet_type_2
    temp = temp
    burnup = burnup
  [../]

  [./fuel_creep]                         # thermal and irradiation creep for UO2 (bison kernel)
    type = CreepUO2
    block = pellet_type_2
    disp_r = disp_x
    disp_z = disp_y
    temp = temp
    fission_rate = fission_rate
    youngs_modulus = 2.e11
    poissons_ratio = .345
    thermal_expansion = 10e-6
    grain_radius = 10.0e-6
    oxy_to_metal_ratio = 2.0
    max_its = 10
    output_iteration_info = false
    stress_free_temperature = 295.0
  [../]

  [./fuel_relocation]
    type = RelocationUO2
    block = pellet_type_2
    burnup = burnup
    diameter = 0.0082
    q = q
    gap = 160e-6 # diametral gap
    burnup_relocation_stop = 1.e20
  [../]



  [./clad_thermal]                       # general thermal property input (elk kernel)
    type = HeatConductionMaterial
    block = clad
    thermal_conductivity = 16.0
    specific_heat = 330.0
  [../]

  [./clad_solid_mechanics] # thermoelasticity and thermal and irradiation creep for Zr4 (bison kernel)
    type = ThermalIrradiationCreepZr4
    block = clad
    disp_r = disp_x
    disp_z = disp_y
    temp = temp
    fast_neutron_flux = fast_neutron_flux
    youngs_modulus = 7.5e10
    poissons_ratio = 0.3
    thermal_expansion = 5.0e-6
    max_its = 5000
    output_iteration_info = false
    stress_free_temperature = 295.0
  [../]

  [./clad_irrgrowth]
    type = IrradiationGrowthZr4
    block = clad
    fast_neutron_fluence = fast_neutron_fluence
  [../]


  [./fission_gas_release]
    type = Sifgrs
    block = pellet_type_2
    temp = temp
    fission_rate = fission_rate        # coupling to fission_rate aux variable
    initial_grain_radius = 10e-6
    grain_radius = grain_radius
    gbs_model = true
  [../]

  [./clad_density]
    type = Density
    block = clad
    density = 6551.0
    disp_r = disp_x
    disp_z = disp_y
  [../]
  [./fuel_density]
    type = Density
    block = pellet_type_2
    disp_r = disp_x
    disp_z = disp_y
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

  type = AdaptiveTransient


  #Preconditioned JFNK (default)
  solve_type = 'PJFNK'


  print_linear_residuals = true
  petsc_options = '-ksp_gmres_modifiedgramschmidt'
  petsc_options_iname = '-snes_linesearch_type -ksp_gmres_restart -pc_type  -pc_composite_pcs -sub_0_pc_hypre_type -sub_0_pc_hypre_boomeramg_max_iter -sub_0_pc_hypre_boomeramg_grid_sweeps_all -sub_1_sub_pc_type -pc_composite_type -ksp_type -mat_mffd_type'
  petsc_options_value = 'basic                   201                 composite hypre,asm         boomeramg            2                                  2                                         lu                 multiplicative     fgmres    ds'


  line_search = 'none'


  l_max_its = 100
  l_tol = 8e-3

  nl_max_its = 15
  nl_rel_tol = 1e-4
  nl_abs_tol = 1e-10

  start_time = -200
  n_startup_steps = 1
  dt = 2.0e2
  end_time = 8.0e7
  num_steps = 5000

  dtmax = 2e6
  dtmin = 1
  optimal_iterations = 6
  iteration_window = 2
  linear_iteration_ratio = 100

  [./Quadrature]
    order = THIRD
  [../]
[]

[Postprocessors]
  # Define postprocessors (some are required as specified above; others are optional; many others are available)

  [./ave_temp_interior]            # average temperature of the cladding interior and all pellet exteriors
     type = SideAverageValue
     boundary = 9
     variable = temp
   [../]

  [./clad_inner_vol]              # volume inside of cladding
   type = InternalVolume
    boundary = 7
    output = file
  [../]

  [./pellet_volume]               # fuel pellet total volume
    type = InternalVolume
    boundary = 8
    output = file
  [../]

  [./avg_clad_temp]               # average temperature of cladding interior
    type = SideAverageValue
    boundary = 7
    variable = temp
  [../]

  [./fis_gas_produced]           # fission gas produced (moles)
    type = ElementIntegralFisGasGeneratedSifgrs
    variable = temp
    block = pellet_type_2
  [../]

  [./fis_gas_released]           # fission gas released to plenum (moles)
    type = ElementIntegralFisGasReleasedSifgrs
    variable = temp
    block = pellet_type_2
  [../]
  [./fis_gas_grain]
    type = ElementIntegralFisGasGrainSifgrs
    variable = temp
    block = pellet_type_2
    output = file
  [../]
  [./fis_gas_boundary]
    type = ElementIntegralFisGasBoundarySifgrs
    variable = temp
    block = pellet_type_2
    output = file
  [../]

  [./gas_volume]                # gas volume
    type = InternalVolume
    boundary = 9
  [../]

  [./plenum_pressure]          # pressure within plenum and gap
    type = Reporter
  [../]

  [./initial_moles]            # initial fill gas mass (moles)
    type = Reporter
    output = file
  [../]

  [./flux_from_clad]           # area integrated heat flux from the cladding
    type = SideFluxIntegral
    variable = temp
    boundary = 5
    diffusivity = thermal_conductivity
  [../]

  [./flux_from_fuel]          # area integrated heat flux from the fuel
    type = SideFluxIntegral
    variable = temp
    boundary = 10
    diffusivity = thermal_conductivity
  [../]

  [./_dt]                     # time step
    type = TimestepSize
  [../]

  [./nonlinear_its]           # number of nonlinear iterations at each timestep
    type = NumNonlinearIterations
  [../]

  [./rod_total_power]
    type = ElementIntegralPower
    variable = temp
    fission_rate = fission_rate
    block = pellet_type_2
  [../]

  [./rod_input_power]
    type = PlotFunction
    function = power_history
    scale_factor = 0.1186 # rod height
  [../]

  [./max_stress]
    type = NodalExtremeValue
    block = clad
    value_type = max
    variable = vonmises
  [../]

[]

[Output]
  # Define output file(s)
  interval = 1
  output_initial = true
  exodus = true
  perf_log = true
  max_pps_rows_screen = 25
  postprocessor_csv = true
[]
