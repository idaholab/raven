[GlobalParams]
  density = 10420
  disp_x = disp_x
  disp_y = disp_y
  disp_r = disp_x
  disp_z = disp_y
  order = SECOND
  family = LAGRANGE
  energy_per_fission = 3.2e-11 # J/fission
[]

[Problem]
  coord_type = RZ
[]

[Mesh]
  file = pk6-3_smeared.e
  displacements = 'disp_x disp_y'
  patch_size = 20
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

[Functions]
  [./power_history]
    type = PiecewiseLinear
    data_file = PK6-3_power_history.csv
    format = columns
    scale_factor = 1.
  [../]
  [./axial_power_factors]
    type = PiecewiseBilinear
    data_file = PK6-3_axial_power_factors.csv
    axis = 1
  [../]
  [./q]
    type = CompositeFunction
    functions = 'power_history axial_power_factors'
  [../]
  [./clad_out_temp]
    type = PiecewiseLinear
    data_file = PK6-3_clad_out_temp.csv
    format = columns
  [../]
  [./axial_temp_factors]
    type = PiecewiseBilinear
    data_file = PK6-3_axial_temp_factors.csv
    axis = 1
  [../]
  [./clad_temp_bc]
    type = CompositeFunction
    functions = 'clad_out_temp axial_temp_factors'
  [../]
  [./coolant_pressure]
    type = PiecewiseLinear
    data_file = PK6-3_coolant_pressure.csv
    format = columns
  [../]
  [./fast_flux]
    type = PiecewiseLinear
    data_file = PK6-3_fast_flux.csv
    format = columns
  [../]
[]

[AuxVariables]
  [./fast_neutron_flux]
    block = pellet_type_1
  [../]
  [./fast_neutron_fluence]
    block = pellet_type_1
  [../]
  [./grain_radius]
    block = pellet_type_1
    initial_condition = 1.7e-05
  [../]
  [./porosity]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
    initial_condition = 0.0626
  [../]
  [./pellet_id]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./scale_thickness]
    order  = CONSTANT
    family = MONOMIAL
  [../]
  [./vonmises_stress]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./hydrostatic_stress]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_xx]
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
  [./strain_zz]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./creep_strain_xx]
    order = CONSTANT
    family = MONOMIAL
    block = clad
  [../]
  [./creep_strain_hoop]
    order = CONSTANT
    family = MONOMIAL
    block = clad
  [../]
  [./creep_strain_mag]
    order = CONSTANT
    family = MONOMIAL
    block = clad
  [../]
  [./gas_gen_3]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./gas_grn_3]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./gas_bdr_3]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./gas_rel_3]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./bbl_bdr_2]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./prs_bbl_bdr]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./prseq_bbl_bdr]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./rad_bbl_bdr]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./GBCoverage]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./sat_coverage]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./eff_diff_coeff]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./deltav_v0_bd] #FIXME unchanged?
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./solid_swell] #old? [./deltav_v0_sl] #FIXME
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./densification] # old?  FIXME [./deltav_v0_dn]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./total_swell] # old? FIXME [./deltav_v0_swe]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./fabrication_porosity]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./gaseous_porosity]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./sintering_porosity]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./thermal_conductivity]
    order = CONSTANT
    family = MONOMIAL
    block = pellet_type_1
  [../]
  [./gap_cond]
    order = CONSTANT
    family = MONOMIAL
  [../]
[]

[AuxKernels]
  [./fast_neutron_flux]
    type = FastNeutronFluxAux
    variable = fast_neutron_flux
    block = pellet_type_1
    function = fast_flux
    execute_on = timestep_begin
  [../]
  [./fast_neutron_fluence]
    type = FastNeutronFluenceAux
    variable = fast_neutron_fluence
    fast_neutron_flux = fast_neutron_flux
    execute_on = timestep_begin
  [../]
  [./grain_radius]
    type = GrainRadiusAux
    block = pellet_type_1
    variable = grain_radius
    temp = temp
    execute_on = linear
  [../]
  [./porosity]
    type = PorosityAuxUO2
    block = pellet_type_1
    variable = porosity
    execute_on = linear
  [../]
  [./pelletid]
    type = PelletIdAux
    block = pellet_type_1
    variable = pellet_id
    a_lower = 0.0155
    a_upper = 0.3305
    number_pellets = 28
    execute_on = timestep_begin #initial
  [../]
  [./scl_thickness]
    type = MaterialRealAux
    variable = scale_thickness
    property = scale_thickness
    boundary = 2
  [../]
  [./vonmises]
    type = MaterialTensorAux
    tensor = stress
    variable = vonmises_stress
    quantity = vonmises
    execute_on = timestep_end
  [../]
  [./hydrostatic_stress]
    type = MaterialTensorAux
    tensor = stress
    variable = hydrostatic_stress
    quantity = hydrostatic
    execute_on = timestep_end
  [../]
  [./stress_xx]
    type = MaterialTensorAux
    tensor = stress
    variable = stress_xx
    index = 0
    execute_on = timestep_end
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
  [./strain_zz]
    type = MaterialTensorAux
    tensor = total_strain
    variable = strain_zz
    index = 2
    execute_on = timestep_end
  [../]
  [./creep_strain_xx]
    type = MaterialTensorAux
    tensor = creep_strain
    variable = creep_strain_xx
    index = 0
    execute_on = timestep_end
  [../]
  [./creep_strain_hoop]
    type = MaterialTensorAux
    tensor = creep_strain
    variable = creep_strain_hoop
    index = 2
    execute_on = timestep_end
  [../]
  [./creep_strain_mag]
    type = MaterialTensorAux
    tensor = creep_strain
    variable = creep_strain_mag
    quantity = plasticstrainmag
    execute_on = timestep_end
  [../]
  [./fggen]
    type = MaterialRealAux
    variable = gas_gen_3
    property = gas_gen_3
  [../]
  [./fggrn]
    type = MaterialRealAux
    variable = gas_grn_3
    property = gas_grn_3
  [../]
  [./fgbdr]
    type = MaterialRealAux
    variable = gas_bdr_3
    property = gas_bdr_3
  [../]
  [./fgrel]
    type = MaterialRealAux
    variable = gas_rel_3
    property = gas_rel_3
  [../]
  [./nbbl2]
    type = MaterialRealAux
    variable = bbl_bdr_2
    property = bbl_bdr_2
  [../]
  [./prsbbl]
    type = MaterialRealAux
    variable = prs_bbl_bdr
    property = prs_bbl_bdr
  [../]
  [./prseqbbl]
    type = MaterialRealAux
    variable = prseq_bbl_bdr
    property = prseq_bbl_bdr
  [../]
  [./radbbl]
    type = MaterialRealAux
    variable = rad_bbl_bdr
    property = rad_bbl_bdr
  [../]
  [./frcvrg]
    type = MaterialRealAux
    variable = GBCoverage
    property = GBCoverage
  [../]
  [./stcvrg]
    type = MaterialRealAux
    variable = sat_coverage
    property = sat_coverage
  [../]
  [./diffc]
    type = MaterialRealAux
    variable = eff_diff_coeff
    property = eff_diff_coeff
  [../]
  [./dvv0bd]
    type = MaterialRealAux
    variable = deltav_v0_bd #FIXME no change?
    property = deltav_v0_bd
  [../]
  [./dvv0sl]
    type = MaterialRealAux
    variable = solid_swell
    property = solid_swell #FIXME deltav_v0_sl
  [../]
  [./dvv0dn]
    type = MaterialRealAux
    variable = densification
    property = densification #FIXME deltav_v0_dn
  [../]
  [./dvv0swe]
    type = MaterialRealAux
    variable = total_swell
    property = total_swell #FIXME deltav_v0_swe
  [../]
  [./fabpor]
    type = MaterialRealAux
    variable = fabrication_porosity
    property = fabrication_porosity
  [../]
  [./gaspor]
    type = MaterialRealAux
    variable = gaseous_porosity
    property = gaseous_porosity
  [../]
  [./sinpor]
    type = MaterialRealAux
    variable = sintering_porosity
    property = sintering_porosity
  [../]
  [./fuel_conductivity]
    type = MaterialRealAux
    variable = thermal_conductivity
    property = thermal_conductivity
  [../]
  [./gap_conductance]
    type = MaterialRealAux
    property = gap_conductance
    variable = gap_cond
    boundary = 10
  [../]
[]

[SolidMechanics]
  [./solid]
    temp = temp
  [../]
[]

[Kernels]
  [./heat]
    type = HeatConduction
    variable = temp
  [../]
  [./heat_ie]
    type = HeatConductionTimeDerivative
    variable = temp
  [../]
  [./heat_source_]
     type = NeutronHeatSource
     variable = temp
     block = pellet_type_1
     fission_rate = fission_rate
  [../]
  [./gravity]
    type = Gravity
    variable = disp_y
    value = -9.81
  [../]
[]

[Burnup]
  [./burnup]
    block = pellet_type_1
    rod_ave_lin_pow = power_history
    axial_power_profile = axial_power_factors
    num_radial = 80
    num_axial = 5
    a_lower = 0.0155
    a_upper = 0.3305
    fuel_inner_radius = 0
    fuel_outer_radius = 0.004573
    i_enrich = ' 0.02985 0.97015 0. 0. 0. 0.'
    RPF = RPF
  [../]
[]

[Contact]
  [./pellet_clad_mechanical]
    master = 5
    slave = 10
    penalty = 1.e+07
    model = frictionless
    normal_smoothing_distance = 0.1
    system = Constraint
  [../]
[]

[ThermalContact]
  [./thermal_contact]
    type = GapHeatTransferLWR
    variable = temp
    master = 5
    slave = 10
    initial_moles = initial_moles
    gas_released = fis_gas_released
    roughness_clad = 0.5e-6
    roughness_fuel = 2.0e-6
    #roughness_coef = 3.2
    plenum_pressure = plenum_pressure
    jump_distance_model = KENNARD
    initial_gas_fractions = '1 0 0 0 0 0 0 0 0 0'
    contact_pressure = contact_pressure
    quadrature = true
    normal_smoothing_distance = 0.1
  [../]
[]

[BCs]
  [./no_x_all]
    type = DirichletBC
    variable = disp_x
    boundary = 12
    value = 0.
  [../]
  [./no_y_clad_bottom]
    type = DirichletBC
    variable = disp_y
    boundary = 1
    value = 0.
  [../]
  [./no_y_fuel_bottom]
    type = DirichletBC
    variable = disp_y
    boundary = 20
    value = 0.
  [../]
  [./temp]
    type = FunctionDirichletBC
    boundary = '1 2 3'
    variable = temp
    function = clad_temp_bc
  [../]
  [./Pressure]
    [./coolantPressure]
      boundary = '1 2 3'
      function = coolant_pressure
    [../]
  [../]
  [./PlenumPressure]
    [./plenumPressure]
      boundary = 9
      initial_pressure = 2.25e+06
      startup_time = 0
      R = 8.3143
      output_initial_moles = initial_moles
      temperature = ave_temp_interior
      volume = gas_volume
      material_input = fis_gas_released
      output = plenum_pressure
    [../]
  [../]
[]

[Materials]
  [./density_fuel]
    type = Density
    block = pellet_type_1
  [../]
  [./fuel_thermal]
    type = ThermalFuel
    block = pellet_type_1
    temp = temp
    burnup = burnup
    model = 4
    initial_porosity = 0.05
  [../]
  [./fuel_solid_mechanics_swelling] #new updated since march 2015
    type = VSwellingUO2
    temp = temp
    burnup_function = burnup
    # density??
    gas_swelling_type = SIFGRS
    block = pellet_type_1
    save_solid_swell = true
    solid_factor = 5.77e-5
    save_densification = true
  [../]
  [./fuel_elastic]
    type = Elastic
    block = pellet_type_1
    temp = temp
    # user manual says I need this....
    dep_matl_props = deltav_v0_bd
    youngs_modulus = 2.e+11
    poissons_ratio = 0.345
    thermal_expansion = 10.e-06
    # volumetric_strain = deltav_v0_swe # does this need to go?
  [../]
  [./fuel_relocation]
    type = RelocationUO2
    block = pellet_type_1
    burnup = burnup
    diameter = 0.009146
    q = q
    gap = 7.3e-05
    burnup_relocation_stop = 0.029
    relocation_activation1 = 5000
  [../]
  [./fission_gas_release_and_swelling]
    type = Sifgrs
    block = pellet_type_1
    # compute_swelling = true # deprecated
    diff_coeff_option = 3
    transient_option = 2
    res_param_option = 1
    temp = temp
    fission_rate = fission_rate
    burnup = burnup
    initial_porosity = 0.05
    # initial_grain_radius = 1.7e-05
    grain_radius = grain_radius
    gbs_model = true
    pellet_id = pellet_id
    pellet_brittle_zone = pbz
    ath_model = true
    rod_ave_lin_pow = power_history
    axial_power_profile = axial_power_factors
    grainradius_scalef = 1.
    igdiffcoeff_scalef = 1.
    resolutionp_scalef = 1.
    # solid_swelling_factor = 5.577e-5
    # total_densification = 0.01
  [../]
  [./density_clad]
    type = Density
    block = clad
    density = 6550.
  [../]
  [./clad_thermal]
    type = HeatConductionMaterial
    block = clad
    thermal_conductivity = 16.
    specific_heat = 330.
  [../]
  [./clad_solid_mechanics]
    type = MechZry
    block = clad
    temp = temp
    fast_neutron_flux = fast_neutron_flux
    fast_neutron_fluence = fast_neutron_fluence
    youngs_modulus = 1.e+11
    poissons_ratio = 0.3
    thermal_expansion = 5.e-06
    absolute_tolerance = 1.e-12
    output_iteration_info = false
    model_irradiation_growth = true
    model_thermal_expansion = false
    max_its = 100
  [../]
  [./clad_oxidation]
    type = OxidationCladding
    boundary = 2
    temperature = temp
    clad_inner_radius = 0.004646
    clad_outer_radius = 0.0053725
    normtemp_model = 0
    hightemp_model = 0
    use_coolant_channel = true
  [../]
[]

[Dampers]
  [./limitT]
    type = MaxIncrement
    max_increment = 50.
    variable = temp
  [../]
[]

[Executioner]
  type = Transient
  solve_type = 'PJFNK'
  line_search = 'none'
  petsc_options = '-snes_ksp_ew'
  petsc_options_iname = '-pc_type  -pc_factor_mat_solver_package  -ksp_gmres_restart'
  petsc_options_value = '      lu                   superlu_dist                 100'

  l_max_its = 100
  l_tol = 1.e-02

  nl_max_its = 10
  nl_rel_tol = 1e-04
  nl_abs_tol = 1e-06

  start_time = -100.
  end_time = 76630068. #7.66e7
  num_steps = 5000.
  dtmax = 1.e+06
  dtmin = 0.1

  [./TimeStepper]
    type = IterationAdaptiveDT
    optimal_iterations = 15
    iteration_window = 3
    linear_iteration_ratio = 25
    growth_factor = 1.2
    cutback_factor = 0.5
    dt = 100.
    time_t =  '0.   76320360. 76324410. 76518360. 76561668. 76626468.'
    time_dt = '100. 100.      1000.     1000.     1000.     1000.    '
  [../]

  [./Quadrature]
    order = FIFTH
    side_order = SEVENTH
  [../]
[]

[UserObjects]
  [./pbz]
   type = PelletBrittleZone
    block = pellet_type_1
    pellet_id = pellet_id
    temp = temp
    pellet_radius = 0.004573
    a_lower = 0.0155
    a_upper = 0.3305
    number_pellets = 28
    execute_on = linear
  [../]
[]

[Postprocessors]
  [./ave_temp_interior]
    type = SideAverageValue
    boundary = 9
    variable = temp
    execute_on = linear
  [../]
  [./clad_inner_vol]
   type = InternalVolume
    boundary = 7
    outputs = exodus
  [../]
  [./pellet_volume]
    type = InternalVolume
    boundary = 8
    outputs = exodus
  [../]
  [./avg_clad_temp]
    type = SideAverageValue
    boundary = 7
    variable = temp
  [../]
  [./fis_gas_generated]
    type = ElementIntegralFisGasGeneratedSifgrs
    variable = temp
    block = pellet_type_1
  [../]
  [./fis_gas_released]
    type = ElementIntegralFisGasReleasedSifgrs
    variable = temp
    block = pellet_type_1
  [../]
  [./gas_volume]
    type = InternalVolume
    boundary = 9
    execute_on = linear
  [../]
  [./average_burnup]
    type = ElementAverageValue
    block = pellet_type_1
    variable = burnup
  [../]
  [./_dt]
    type = TimestepSize
  [../]
  [./nonlinear_its]
    type = NumNonlinearIterations
  [../]
  [./max_fuel_temp]
    type = NodalExtremeValue
    block = pellet_type_1
    value_type = max
    variable = temp
  [../]
  [./centerline_temp]
     type = NodalVariableValue
     variable = temp
     nodeid = 1256
  [../]
  [./max_clad_temp]
    type = NodalExtremeValue
    block = clad
    value_type = max
    variable = temp
  [../]
  [./midplane_clad_temp]
    type = NodalVariableValue
    nodeid = 319
    variable = temp
  [../]
  [./avg_gap_conductance]
    type = SideAverageValue
    boundary = 10
    variable = gap_cond
  [../]
  [./max_contact_pressure]
    type = ElementExtremeValue
    value_type = max
    variable = contact_pressure
  [../]
  [./midplane_contact_pressure]
    type = ElementalVariableValue
    elementid = 352
    variable = contact_pressure
  [../]
  [./max_hoop_stress]
    type = ElementExtremeValue
    block = clad
    value_type = max
    variable = stress_zz
  [../]
  [./max_vonmises_stress]
    type = ElementExtremeValue
    block = clad
    value_type = max
    variable = vonmises_stress
  [../]
  [./midplane_hoop_stress]
    type = ElementalVariableValue
    elementid = 89
    variable = stress_zz
  [../]
  [./midplane_vonmises_stress]
    type = ElementalVariableValue
    elementid = 89
    variable = vonmises_stress
  [../]
  [./max_hoop_strain]
    type = ElementExtremeValue
    block = clad
    value_type = max
    variable = strain_zz
  [../]
  [./midplane_hoop_strain]
    type = ElementalVariableValue
    elementid = 89
    variable = strain_zz
  [../]
  [./max_radial_disp]
    type = NodalExtremeValue
    boundary = 2
    value_type = max
    variable = disp_x
  [../]
  [./midplane_radial_disp]
    type = NodalVariableValue
    nodeid = 309
    variable = disp_x
  [../]
  [./max_oxide_thickness]
    type = ElementExtremeValue
    value_type = max
    variable = scale_thickness
  [../]
  [./midplane_oxide_thickness]
    type = ElementalVariableValue
    elementid = 86
    variable = scale_thickness
  [../]
[]

# Define output file(s)
[Outputs]
  checkpoint = true
  file_base = output_pk63
  interval = 1
  csv = true
  exodus = true
  color = false
  sync_times = '0 1560 5400 8514 10800 13920 21600 24714 25200 27852 3949200 3951036 3952800 3953550 6465600 6465972 6469200 6469578 9190800 9191100 9194400 9194694 11444400 11444664 11448000 11448264 13698000 13698186 13701600 13701792 16221600 16221642 16225200 16225248 18579600 18579666 18583200 18583260 21304800 21305220 21308400 21308820 24030000 24030078 24033600 24033678 25473600 25474176 25477200 25477776 27604800 27608400 30168000 30168018 30171600 30171612 32752800 32753034 32756400 32756634 35100000 35100078 35103600 35103678 37105200 37108800 39758400 39758508 39762000 39762114 42206400 42210000 44748000,44748060,44751600,44751660,47379600,47380068,47383200,47383668,49460400,49460868,49464000,49464468,51044400,51044430,51048000,51048036,53816400,53816742 53820000 53820342 56286000 56286018 56289600 56289612 58888800 58888926 58892400 58892526 61732800 61732908 61736400 61736508 63370800 63370812 63374400 63374520 66056400 66056760 66060000 66060216 68641200 68641266 68644800 68644860 70891200 71557200 71559288 76320000 76320360 76324410 76406760 76410360 76417860 76431960 76439460 76518360 76518480 76561668 76574568 76583268 76596168 76626468 76630068'
  [./console]
    type = Console
    perf_log = true
    max_rows = 25
    output_linear = true
  [../]
[]

#[Debug]
#  show_var_residual = 'disp_x disp_y temp'
#  show_var_residual_norms = true
#[]
