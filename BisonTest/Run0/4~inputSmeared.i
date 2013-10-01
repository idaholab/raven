[GlobalParams]
  # Set initial fuel density, other global parameters
  disp_y =  disp_y
  family =  LAGRANGE
  disp_x =  disp_x
  density =  10431.0
  energy_per_fission =  3.2e-11  # J/fission
  order =  SECOND
[]
[Problem]
  # Specify coordinate system type
  coord_type =  RZ
[]
[Mesh]
  # Import mesh file
  patch_size =  1000 # For contact algorithm
  displacements =  'disp_x disp_y'
  file =  smeared.e
[]
[Variables]
  # Define dependent variables and initial conditions
  [./disp_x]
  [../]
  [./disp_y]
  [../]
  [./temp]
    initial_condition =  580.0     # set initial temp to coolant inlet
  [../]
[]
[AuxVariables]
  # Define auxilary variables
  [./fast_neutron_flux]
    block =  clad
  [../]
  [./fast_neutron_fluence]
    block =  clad
  [../]
  [./grain_radius]
    block =  pellet_type_2
    initial_condition =  10e-6
  [../]
  [./stress_xx]      # stress aux variables are defined for output; this is a way to get integration point variables to the output file]
    order =  CONSTANT
    family =  MONOMIAL
  [../]
  [./stress_yy]
    order =  CONSTANT
    family =  MONOMIAL
  [../]
  [./stress_zz]
    order =  CONSTANT
    family =  MONOMIAL
  [../]
  [./vonmises]
    order =  CONSTANT
    family =  MONOMIAL
  [../]
  [./creep_strain_mag]
    order =  CONSTANT
    family =  MONOMIAL
  [../]
  [./gap_cond]
    order =  CONSTANT
    family =  MONOMIAL
  [../]
  [./coolant_htc]
    order =  CONSTANT
    family =  MONOMIAL
  [../]
[]
[Functions]
  # Define functions to control power and boundary conditions
  [./power_history]
    scale_factor =  1
    type =  PiecewiseLinearFile   # reads and interpolates an input file containing rod average linear power vs time
    value = 1.00217634087
    yourFileName =  powerhistory.csv
  [../]
  [./axial_peaking_factors]      # reads and interpolates an input file containing the axial power profile vs time]
    scale_factor =  1
    type =  PiecewiseBilinear
    yourFileName =  peakingfactors12.csv
    axis =  1 # (0,1,2) 
  [../]
  [./pressure_ramp]              # reads and interpolates input data defining amplitude curve for fill gas pressure]
    y =  '0 1'
    x =  '-200 0'
    type =  PiecewiseLinear
  [../]
  [./q]
    functions =  'power_history axial_peaking_factors'
    type =  CompositeFunction
  [../]
[]
[SolidMechanics]
  # Specify that we need solid mechanics (divergence of stress)
  [./solid]
    disp_r =  disp_x
    temp =  temp
    disp_z =  disp_y
  [../]
[]
[Kernels]
  # Define kernels for the various terms in the PDE system
  [./gravity]       # body force term in stress equilibrium equation]
    variable =  disp_y
    type =  Gravity
    value =  -9.81
  [../]
  [./heat]         # gradient term in heat conduction equation]
    variable =  temp
    type =  HeatConduction
  [../]
  [./heat_ie]       # time term in heat conduction equation]
    variable =  temp
    type =  HeatConductionTimeDerivative
  [../]
  [./heat_source]  # source term in heat conduction equation]
    variable =  temp
    type =  NeutronHeatSource
    fission_rate =  fission_rate  # coupling to the fission_rate aux variable
    block =  pellet_type_2     # fission rate applied to the fuel (block 2) only
  [../]
[]
[Burnup]
  [./burnup]
    #N235 = N235 # Activate to write N235 concentration to output file
    #N238 = N238 # Activate to write N238 concentration to output file
    #N239 = N239 # Activate to write N239 concentration to output file
    #N240 = N240 # Activate to write N240 concentration to output file
    #N241 = N241 # Activate to write N241 concentration to output file
    #N242 = N242 # Activate to write N242 concentration to output file
    fuel_outer_radius =  .0041
    axial_power_profile =  axial_peaking_factors     # using the axial power profile function defined above
    num_axial =  11
    fuel_volume_ratio =  1.0 # for use with dished pellets (ratio of actual volume to cylinder volume)
    a_upper =  0.12804   # mesh dependent
    RPF =  RPF
    num_radial =  80
    rod_ave_lin_pow =  power_history          # using the power function defined above
    a_lower =  0.00944   # mesh dependent
    block =  pellet_type_2
    fuel_inner_radius =  0
  [../]
[]
[AuxKernels]
  # Define auxilliary kernels for each of the aux variables
  [./fast_neutron_flux]
    rod_ave_lin_pow =  power_history
    execute_on =  timestep_begin
    factor =  3e13
    variable =  fast_neutron_flux
    axial_power_profile =  axial_peaking_factors
    type =  FastNeutronFluxAux
    block =  clad
  [../]
  [./fast_neutron_fluence]
    variable =  fast_neutron_fluence
    fast_neutron_flux =  fast_neutron_flux
    type =  FastNeutronFluenceAux
    block =  clad
    execute_on =  timestep_begin
  [../]
  [./grain_radius]
    variable =  grain_radius
    execute_on =  residual
    type =  GrainRadiusAux
    temp =  temp
    block =  pellet_type_2
  [../]
  [./stress_xx]               # computes stress components for output]
    variable =  stress_xx
    index =  0
    type =  MaterialTensorAux
    tensor =  stress
    execute_on =  timestep     # for efficiency, only compute at the end of a timestep
  [../]
  [./stress_yy]
    variable =  stress_yy
    index =  1
    type =  MaterialTensorAux
    tensor =  stress
    execute_on =  timestep
  [../]
  [./stress_zz]
    variable =  stress_zz
    index =  2
    type =  MaterialTensorAux
    tensor =  stress
    execute_on =  timestep
  [../]
  [./vonmises]
    variable =  vonmises
    execute_on =  timestep
    type =  MaterialTensorAux
    tensor =  stress
    quantity =  vonmises
  [../]
  [./creep_strain_mag]
    tensor =  creep_strain
    variable =  creep_strain_mag
    execute_on =  timestep
    type =  MaterialTensorAux
    block =  'clad pellet_type_2'
    quantity =  plasticstrainmag
  [../]
[]
[AuxBCs]
  [./conductance]
    variable =  gap_cond
    boundary =  10
    property =  gap_conductance
    type =  MaterialRealAux
  [../]
  [./coolant_htc]
    variable =  coolant_htc
    boundary =  '1 2 3'
    property =  coolant_channel_htc
    type =  MaterialRealAux
  [../]
[]
[Contact]
  # Define mechanical contact between the fuel (sideset=10) and the clad (sideset=5)
  [./pellet_clad_mechanical]
    penalty =  1e7
    slave =  10
    master =  5
    disp_y =  disp_y
    disp_x =  disp_x
  [../]
[]
[ThermalContact]
  # Define thermal contact between the fuel (sideset=10) and the clad (sideset=5)
  [./thermal_contact]
    slave =  8
    initial_moles =  initial_moles       # coupling to a postprocessor which supplies the initial plenum/gap gas mass
    quadrature =  true
    master =  7
    gas_released =  fis_gas_released     # coupling to a postprocessor which supplies the fission gas addition
    variable =  temp
    contact_pressure =  contact_pressure
    type =  GapHeatTransferLWR
  [../]
  [./thermal_contact_bottom]
    slave =  21
    initial_moles =  initial_moles       # coupling to a postprocessor which supplies the initial plenum/gap gas mass
    quadrature =  true
    master =  22
    gas_released =  fis_gas_released     # coupling to a postprocessor which supplies the fission gas addition
    variable =  temp
    contact_pressure =  contact_pressure
    type =  GapHeatTransferLWR
  [../]
  [./thermal_contact_top]
    slave =  24
    initial_moles =  initial_moles       # coupling to a postprocessor which supplies the initial plenum/gap gas mass
    quadrature =  true
    master =  23
    gas_released =  fis_gas_released     # coupling to a postprocessor which supplies the fission gas addition
    variable =  temp
    contact_pressure =  contact_pressure
    type =  GapHeatTransferLWR
  [../]
[]
[BCs]
  # Define boundary conditions
  [./no_x_all] # pin pellets and clad along axis of symmetry (y)]
    variable =  disp_x
    boundary =  12
    type =  DirichletBC
    value =  0.0
  [../]
  [./no_y_clad_bottom] # pin clad bottom in the axial direction (y)]
    variable =  disp_y
    boundary =  '1'
    type =  DirichletBC
    value =  0.0
  [../]
  [./no_y_fuel_bottom] # pin fuel bottom in the axial direction (y)]
    variable =  disp_y
    boundary =  '1020'
    type =  DirichletBC
    value =  0.0
  [../]
  [./Pressure] #  apply coolant pressure on clad outer walls]
  [../]
  [./PlenumPressure] #  apply plenum pressure on clad inner walls and pellet surfaces]
  [../]
[]
[CoolantChannel]
  [./convective_clad_surface] # apply convective boundary to clad outer surface]
    inlet_pressure =  15.5e6   # Pa
    rod_diameter =  0.948e-2 # m
    inlet_massflux =  3800     # kg/m^2-sec
    rod_pitch =  1.26e-2  # m
    linear_heat_rate =  power_history
    variable =  temp
    axial_power_profile =  axial_peaking_factors
    boundary =  '1 2 3'
    inlet_temperature =  580      # K
  [../]
[]
[PlenumTemperature]
  [./plenumTemp]
    temp =  temp
    num_pellets =  3
  [../]
[]
[Materials]
  # Define material behavior models and input material property data
  [./hafnium_thermal]
    specific_heat =  286.5
    type =  HeatConductionMaterial
    block =  'pellet_type_1 pellet_type_3'
    thermal_conductivity =  2.0
  [../]
  [./hafnium_density]
    disp_z =  disp_y
    type =  Density
    disp_r =  disp_x
    block =  'pellet_type_1 pellet_type_3'
    density =  9680.0
  [../]
  [./hafnium_elastic]
    stress_free_temperature =  295.0
    disp_z =  disp_y
    temp =  temp
    poissons_ratio =  0.26
    disp_r =  disp_x
    thermal_expansion =  1.4732e-07
    youngs_modulus =  130e9
    type =  Elastic
    block =  'pellet_type_1 pellet_type_3'
  [../]
  [./fuel_thermal]                       # temperature and burnup dependent thermal properties of UO2 (bison kernel)]
    burnup =  burnup
    type =  ThermalUO2
    temp =  temp
    block =  pellet_type_2
  [../]
  [./fuel_solid_mechanics_swelling]      # free expansion strains (swelling and densification) for UO2 (bison kernel)]
    burnup =  burnup
    type =  VSwellingUO2
    temp =  temp
    block =  pellet_type_2
  [../]
  [./fuel_creep]                         # thermal and irradiation creep for UO2 (bison kernel)]
    stress_free_temperature =  295.0
    grain_radius =  10.0e-6
    oxy_to_metal_ratio =  2.0
    temp =  temp
    fission_rate =  fission_rate
    poissons_ratio =  .345
    disp_r =  disp_x
    output_iteration_info =  false
    thermal_expansion =  10e-6
    youngs_modulus =  2.e11
    type =  CreepUO2
    max_its =  10
    block =  pellet_type_2
    disp_z =  disp_y
  [../]
  [./fuel_relocation]
    diameter =  0.0082
    gap =  160e-6 # diametral gap
    q =  q
    block =  pellet_type_2
    burnup_relocation_stop =  1.e20
    type =  RelocationUO2
    burnup =  burnup
  [../]
  [./clad_thermal]                       # general thermal property input (elk kernel)]
    specific_heat =  330.0
    type =  HeatConductionMaterial
    block =  clad
    thermal_conductivity =  16.0
  [../]
  [./clad_solid_mechanics]               # thermoelasticity and thermal and irradiation creep for Zr4 (bison kernel)]
    stress_free_temperature =  295.0
    disp_z =  disp_y
    temp =  temp
    poissons_ratio =  0.3
    disp_r =  disp_x
    output_iteration_info =  false
    thermal_expansion =  5.0e-6
    fast_neutron_flux =  fast_neutron_flux
    youngs_modulus =  7.5e10
    type =  ThermalIrradiationCreepZr4
    max_its =  5000
    block =  clad
  [../]
  [./clad_irrgrowth]
    fast_neutron_fluence =  fast_neutron_fluence
    type =  IrradiationGrowthZr4
    block =  clad
  [../]
  [./fission_gas_release]
    grain_radius =  grain_radius
    temp =  temp
    fission_rate =  fission_rate        # coupling to fission_rate aux variable
    initial_grain_radius =  10e-6
    value = 6.89365392202e-06
    gbs_model =  true
    type =  Sifgrs
    block =  pellet_type_2
  [../]
  [./clad_density]
    disp_z =  disp_y
    type =  Density
    disp_r =  disp_x
    block =  clad
    density =  6551.0
  [../]
  [./fuel_density]
    type =  Density
    disp_r =  disp_x
    block =  pellet_type_2
    disp_z =  disp_y
  [../]
  [./clad_solid_mechanics]
    value = 12779181996.3
  [../]
[]
[Dampers]
  [./limitT]
    variable =  temp
    type =  MaxIncrement
    max_increment =  100.0
  [../]
  [./limitX]
    variable =  disp_x
    type =  MaxIncrement
    max_increment =  1e-5
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
  nl_abs_tol =  1e-10
  nl_rel_tol =  1e-4
  print_linear_residuals =  true
  num_steps =  5000
  line_search =  'none'
  iteration_window =  2
  petsc_options =  '-ksp_gmres_modifiedgramschmidt'
  n_startup_steps =  1
  type =  AdaptiveTransient
  petsc_options_value =  'basic                   201                 composite hypre,asm         boomeramg            2                                  2                                         lu                 multiplicative     fgmres    ds'
  linear_iteration_ratio =  100
  l_max_its =  100
  start_time =  -200
  dtmax =  2e6
  nl_max_its =  15
  dt =  2.0e2
  petsc_options_iname =  '-snes_linesearch_type -ksp_gmres_restart -pc_type  -pc_composite_pcs -sub_0_pc_hypre_type -sub_0_pc_hypre_boomeramg_max_iter -sub_0_pc_hypre_boomeramg_grid_sweeps_all -sub_1_sub_pc_type -pc_composite_type -ksp_type -mat_mffd_type'
  optimal_iterations =  6
  l_tol =  8e-3
  solve_type =  'PJFNK'
  end_time =  8.0e7
  dtmin =  1
  [./Quadrature]
    order =  THIRD
  [../]
[]
[Postprocessors]
  # Define postprocessors (some are required as specified above; others are optional; many others are available)
  [./ave_temp_interior]            # average temperature of the cladding interior and all pellet exteriors]
    variable =  temp
    boundary =  9
    type =  SideAverageValue
  [../]
  [./clad_inner_vol]              # volume inside of cladding]
    output =  file
    boundary =  7
    type =  InternalVolume
  [../]
  [./pellet_volume]               # fuel pellet total volume]
    output =  file
    boundary =  8
    type =  InternalVolume
  [../]
  [./avg_clad_temp]               # average temperature of cladding interior]
    variable =  temp
    boundary =  7
    type =  SideAverageValue
  [../]
  [./fis_gas_produced]           # fission gas produced (moles)]
    variable =  temp
    type =  ElementIntegralFisGasGeneratedSifgrs
    block =  pellet_type_2
  [../]
  [./fis_gas_released]           # fission gas released to plenum (moles)]
    variable =  temp
    type =  ElementIntegralFisGasReleasedSifgrs
    block =  pellet_type_2
  [../]
  [./fis_gas_grain]
    variable =  temp
    output =  file
    type =  ElementIntegralFisGasGrainSifgrs
    block =  pellet_type_2
  [../]
  [./fis_gas_boundary]
    variable =  temp
    output =  file
    type =  ElementIntegralFisGasBoundarySifgrs
    block =  pellet_type_2
  [../]
  [./gas_volume]                # gas volume]
    boundary =  9
    type =  InternalVolume
  [../]
  [./plenum_pressure]          # pressure within plenum and gap]
    type =  Reporter
  [../]
  [./initial_moles]            # initial fill gas mass (moles)]
    output =  file
    type =  Reporter
  [../]
  [./flux_from_clad]           # area integrated heat flux from the cladding]
    variable =  temp
    boundary =  5
    diffusivity =  thermal_conductivity
    type =  SideFluxIntegral
  [../]
  [./flux_from_fuel]          # area integrated heat flux from the fuel]
    variable =  temp
    boundary =  10
    diffusivity =  thermal_conductivity
    type =  SideFluxIntegral
  [../]
  [./_dt]                     # time step]
    type =  TimestepSize
  [../]
  [./nonlinear_its]           # number of nonlinear iterations at each timestep]
    type =  NumNonlinearIterations
  [../]
  [./rod_total_power]
    variable =  temp
    type =  ElementIntegralPower
    block =  pellet_type_2
    fission_rate =  fission_rate
  [../]
  [./rod_input_power]
    function =  power_history
    scale_factor =  0.1186 # rod height
    type =  PlotFunction
  [../]
  [./max_stress]
    variable =  vonmises
    value_type =  max
    type =  NodalExtremeValue
    block =  clad
  [../]
[]
[Output]
  # Define output file(s)
  exodus =  true
  max_pps_rows_screen =  25
  interval =  1
  postprocessor_csv =  true
  output_initial =  true
  perf_log =  true
[]
