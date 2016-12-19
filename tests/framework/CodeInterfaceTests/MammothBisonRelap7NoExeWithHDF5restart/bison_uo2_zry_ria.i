[GlobalParams]
  family = LAGRANGE
  disp_z = disp_y
  disp_y = disp_y
  disp_x = disp_x
  density = 10431.0
  disp_r = disp_x
  order = SECOND
[]
[Problem]
  solution_variables = 'disp_x disp_y temp'
  acceptable_iterations =  10
  reference_residual_variables = 'saved_x saved_y saved_t'
  acceptable_multiplier =  10
  coord_type = RZ
  type = ReferenceResidualProblem
[]
[Mesh]
  pellet_outer_radius = 0.004096
  type = SmearedPelletMesh
  pellet_quantity = 1
  patch_update_strategy = auto
  top_bot_clad_height = 0.00224
  clad_bot_gap_height = 0.001
  patch_size = 30
  displacements = 'disp_x disp_y'
  clad_mesh_density = customize
  clad_thickness = 0.000579548208099
  elem_type = QUAD8
  ny_cu = 3
  pellet_mesh_density = customize
  ny_p = 500
  clad_gap_width = 3.09471214861e-05
  nx_p = 11
  clad_top_gap_height = 0.16243
  ny_c = 500
  pellet_height = 4.2672
  ny_cl = 3
  bx_p = 1.0
  nx_c = 4
[]
[UserObjects]
  #  [./post_chf_ria_failure]
  #    type = Terminator
  #    expression = 'max_alpha_vapor > 0.45'
  #  [../]
  [./pin_geometry]
    clad_inner_wall = 5
    clad_bottom = 1
    pellet_exteriors = 8
    clad_top = 3
    clad_outer_wall = 2
    type = FuelPinGeometry
  [../]
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
[AuxVariables]
  # from RELAP-7
  [./fast_neutron_flux]
    block = '1'
  [../]
  [./fast_neutron_fluence]
    block = '1'
  [../]
  [./stress_hoop]
    # stress aux variables are defined for output; this is a way to get integration point variables to the output file
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./vonmises]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./stress_axial]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./total_hoop_strain]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./total_axial_strain]
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
  [./creep_strain_mag]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./plastic_strain_mag]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./saved_x]
  [../]
  [./saved_y]
  [../]
  [./saved_t]
  [../]
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
    initial_condition = 1
  [../]
  [./Hw_liquid]
    initial_condition = 1
  [../]
  [./pressure_mix]
    initial_condition = 15.6e6
  [../]
[]
[Functions]
  # RELAP-7 replaced function
  #  [./clad_bc]
  #    type = PiecewiseBilinear
  #    data_file = clad_bc_u3si2.csv
  #    axis = 1
  #  [../]
  #  [./fast_flux_history]
  #    type = ConstantFunction
  #    value =1e17
  #  [../]
  # RELAP-7 replaced function
  #  [./pressure_ramp]              # inlet coolant pressure evolution
  #    type = PiecewiseLinear
  #    x = '-50 0 1'
  #    y = '6.537e-3 1 1'
  #    scale_factor = 1
  #  [../]
  [./power_history]
    #    data_file = power_history.csv
    scale_factor = 1
    xy_data = '0.000000000000000000e+00     0.000000000000000000e+00
1.000000000000000000e+02     1.542215972999999940e+04
2.000000000000000000e+02     1.542215972999999940e+04
2.000099999999999909e+02     2.030610025508576655e+04
2.000240000000000009e+02     3.096706315933872247e+04
2.000420000000000016e+02     5.120081380004575476e+04
2.000540000000000020e+02     7.165654069296710077e+04
2.000729999999999791e+02     1.275464555827322911e+05
2.000840000000000032e+02     1.883478395443792688e+05
2.000939999999999941e+02     2.702932552310064202e+05
2.000999999999999943e+02     3.558695147039041040e+05
2.001069999999999993e+02     4.333520303413145593e+05
2.001100000000000136e+02     4.460718662493329030e+05
2.001119999999999948e+02     4.500000000000000000e+05
2.001140000000000043e+02     4.497840044955831254e+05
2.001159999999999854e+02     4.459202337974691181e+05
2.001179999999999950e+02     4.389749234243460232e+05
2.001279999999999859e+02     3.779870274248858914e+05
2.001399999999999864e+02     3.006828260890996899e+05
2.001519999999999868e+02     2.487030602197604021e+05
2.001639999999999873e+02     2.169404240561699844e+05
2.001879999999999882e+02     1.806502264057414141e+05
2.002119999999999891e+02     1.584417355526646716e+05
2.002500000000000000e+02     1.337636909721966367e+05
2.002839999999999918e+02     1.182450919327925658e+05
2.003160000000000025e+02     1.069746215769547998e+05
2.003619999999999948e+02     9.377068548070246470e+04
2.004879999999999995e+02     6.983683113849229994e+04
2.006119999999999948e+02     5.528341802637941146e+04
2.008000000000000114e+02     3.905522822545759846e+04
2.010000000000000000e+02     2.484226152892134269e+04
5.010000000000000000e+02     1.542215972999999940e+04'
    type = PiecewiseLinear
    format = columns
  [../]
  [./axial_peaking_factors]
    #    data_file = axial_peaking_factors.csv
    y = '0 100 501'
    x = '0.1068 0.3203 0.5338 0.7472 0.9607 1.1742 1.3878 1.6013 1.8148 2.0283 2.2418 2.4553 2.6688 2.8823 3.0958 3.3093 3.5228 3.7363 3.9498 4.1633'
    z = '1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.544 0.795 0.933 1.012 1.065 1.103 1.131 1.151 1.163 1.169 1.168 1.161 1.147 1.126 1.096 1.056 0.998 0.909 0.758 0.516 0.544 0.795 0.933 1.012 1.065 1.103 1.131 1.151 1.163 1.169 1.168 1.161 1.147 1.126 1.096 1.056 0.998 0.909 0.758 0.516'
    type = PiecewiseBilinear
    axis = 1
  [../]
  [./YM_function]
    #Youngs Modulus as a function of temp from PNNL report correlations
    #Cold work = 0.5, Average oxygen concentration = 0, fast neutron fluence = 0
    scale_factor = 1.0
    xy_data = '    0          79375000000
300        79375000000
320        78280000000
340        77185000000
360        76090000000
380        74995000000
400        73900000000
420        72805000000
440        71710000000
460        70615000000
480        69520000000
500        68425000000
520        67330000000
540        66235000000
560        65140000000
580        64045000000
600        62950000000
620        61855000000
640        60760000000
660        59665000000
680        58570000000
700        57475000000
720        56380000000
740        55285000000
760        54190000000
780        53095000000
800        52000000000
820        50905000000
840        49810000000
860        48715000000
880        47620000000
900        46525000000
920        45430000000
940        44335000000
960        43240000000
980        42145000000
1000       41050000000
1020       39955000000
1040       38860000000
1060       37765000000
1080       36670000000
1090       36122500000
1100       36434621212
1120       37058863636
1140       37683106061
1160       38307348485
1180       38931590909
1200       39555833333
1220       40180075758
1240       40804318182
1250       41116439394
1255       41272500000
1260       41070000000
1280       40260000000
1300       39450000000
1320       38640000000
1340       37830000000
1360       37020000000
1380       36210000000
1400       35400000000'
    type = PiecewiseLinear
    format = columns
  [../]
  [./Fuel_CTE_function]
    #Fuel CTE as a function of temp from MATPRO data
    scale_factor = 1.0
    xy_data = '    0          1.00005E-05
300        1.00005E-05
350        1.00005E-05
400        1.00015E-05
450        0.000010004
500        1.00091E-05
550        0.000010018
600        1.00321E-05
650        1.00522E-05
700        1.00791E-05
750        1.01131E-05
800        1.01544E-05
850        1.02028E-05
900        1.02577E-05
950        1.03187E-05
1000       0.000010385
1050       0.000010456
1100       1.05308E-05
1150       1.06087E-05
1200       1.06891E-05
1250       1.07712E-05
1300       1.08545E-05
1350       1.09384E-05
1400       1.10224E-05
1450       1.11061E-05
1500       1.11891E-05
1550       1.12712E-05
1600       1.13519E-05
1650       1.14311E-05
1700       1.15087E-05
1750       1.15844E-05
1800       0.000011658
1850       1.17297E-05
1900       1.17991E-05
1950       1.18664E-05
2000       1.19314E-05
2050       1.19942E-05
2100       1.20547E-05
2150       0.000012113
2200       1.21691E-05
2250       1.22229E-05
2300       1.22746E-05
2350       1.23242E-05
2400       1.23717E-05'
    type = PiecewiseLinear
    format = columns
  [../]
  [./Clad_CTE_function]
    #Cladding CTE as a function of temp from MATPRO data
    scale_factor = 1.0
    xy_data = '    0          0.00000495
300        0.00000495
350        0.00000495
400        0.00000495
450        0.00000495
500        0.00000495
550        0.00000495
600        0.00000495
650        0.00000495
700        0.00000495
750        0.00000495
800        0.00000495
850        0.00000495
900        0.00000495
950        0.00000495
1000       0.00000495
1050       0.00000495
1083       0.00000495
1100       4.76997E-06
1150       3.60423E-06
1200       2.2888E-06
1244       1.77904E-06
1250       1.81579E-06
1300       0.00000221
1350       2.56667E-06
1400       2.89091E-06
1450       3.18696E-06
1500       3.45833E-06
1550       0.000003708
1600       3.93846E-06
1650       4.15185E-06
1700       0.00000435
1750       4.53448E-06
1800       4.70667E-06
1850       4.86774E-06
1900       5.01875E-06
1950       5.16061E-06
2000       5.29412E-06
2050       0.00000542
2098       5.53426E-06'
    type = PiecewiseLinear
    format = columns
  [../]
  [./YS_function]
    #Yield Strength as a function of temp from PNNL report
    #Cold work = 0.5, Average oxygen concentration = 0, fast neutron fluence = 0
    #Assumed constant strain rate of 0.01
    scale_factor = 1.0
    xy_data = '    0          593425301.8
300        593425301.8
320        580674456.3
340        567078477.7
360        552695674.7
380        537584819.6
400        521805085.5
420        505428587.7
440        489716889.1
460        475582155.5
480        462727514.7
500        450878059.4
520        439778204.9
540        429188702.8
560        418888211.8
580        408677370.3
600        398362281.7
620        387769321.2
640        376741767.7
660        365140974.9
680        352847834.1
700        339764457.8
720        325816004.1
740        310952563.6
760        271786784.1
780        222194537.8
800        185377726.5
820        154173430.2
840        130199325.7
860        111396817.2
880        96362607.14
900        84120129.15
920        73977044.17
940        65434357.46
960        58127235.93
980        51785717.25
1000       46208150.84
1020       41242936.22
1040       36775756.82
1060       32720505.36
1080       29012718.16
1090       27273687.16
1100       26355024.23
1120       22442026.84
1140       18753365.2
1160       15296510.38
1180       12079785.07
1200       9113113.474
1220       6409493.205
1240       3988344.555
1250       2893569.671
1255       2377978.33
1260       2323366.884
1280       2116912.621
1300       1928475.009
1320       1756577.201
1340       1599841.384
1360       1456985.125
1380       1326817.541
1400       1208235.326'
    type = PiecewiseLinear
    format = columns
  [../]
[]
[SolidMechanics]
  [./solid]
    save_in_disp_r = saved_x
    save_in_disp_z = saved_y
    temp = temp
  [../]
[]
[Kernels]
  [./heat]
    # gradient term in heat conduction equation
    variable = temp
    save_in = saved_t
    type = HeatConduction
  [../]
  [./heat_ie]
    # time term in heat conduction equation
    variable = temp
    save_in = saved_t
    type = HeatConductionTimeDerivative
  [../]
  [./heat_source_fuel]
    # source term in heat conduction equation
    variable = temp
    save_in = saved_t
    type = NeutronHeatSource
    burnup_function = burnup
    block = '3'
  [../]
[]
[Burnup]
  [./burnup]
    #    fuel_volume_ratio = 0.987775 # for use with dished pellets (ratio of actual volume to cylinder volume)
    axial_power_profile =  axial_peaking_factors
    family = MONOMIAL
    num_axial = 11
    fuel_pin_geometry = pin_geometry
    RPF = RPF
    num_radial = 80
    rod_ave_lin_pow =  power_history
    order = CONSTANT
    block = 3
  [../]
[]
[AuxKernels]
  [./fast_neutron_flux]
    #    function = fast_flux_history
    rod_ave_lin_pow = power_history
    execute_on = timestep_begin
    factor =  3e13
    variable = fast_neutron_flux
    axial_power_profile = axial_peaking_factors
    type = FastNeutronFluxAux
    block = '1'
  [../]
  [./fast_neutron_fluence]
    variable = fast_neutron_fluence
    fast_neutron_flux = fast_neutron_flux
    type = FastNeutronFluenceAux
    block = '1'
    execute_on = timestep_begin
  [../]
  [./stress_hoop]
    # computes stress components for output
    variable = stress_hoop
    execute_on =  timestep_end
    type = MaterialTensorAux
    tensor = stress
    quantity = Hoop
  [../]
  [./stress_axial]
    variable = stress_axial
    execute_on = timestep_end
    type = MaterialTensorAux
    tensor = stress
    quantity = Axial
  [../]
  [./vonmises]
    variable = vonmises
    execute_on = timestep_end
    type = MaterialTensorAux
    tensor = stress
    quantity = vonmises
  [../]
  [./total_hoop_strain]
    # computes stress components for output
    tensor = total_strain
    variable = total_hoop_strain
    execute_on =  timestep_end
    type = MaterialTensorAux
    block = '1'
    quantity = Hoop
  [../]
  [./total_axial_strain]
    # computes stress components for output
    tensor = total_strain
    variable = total_axial_strain
    execute_on =  timestep_end
    type = MaterialTensorAux
    block = '1'
    quantity = Axial
  [../]
  [./creep_strain_mag]
    variable = creep_strain_mag
    execute_on = timestep_end
    type = MaterialTensorAux
    tensor = creep_strain
    quantity = plasticstrainmag
  [../]
  [./plastic_strain_mag]
    variable = plastic_strain_mag
    execute_on = timestep_end
    type = MaterialTensorAux
    tensor = plastic_strain
    quantity = plasticstrainmag
  [../]
  [./conductance]
    variable = gap_cond
    boundary = 10
    property = gap_conductance
    type = MaterialRealAux
  [../]
[]
[Contact]
  [./pellet_clad_mechanical]
    slave = 10
    system = Constraint
    order = SECOND
    penalty = 1e14
    normalize_penalty = true
    master = 5
    model = frictionless
    normal_smoothing_distance = 0.1
  [../]
[]
[ThermalContact]
  [./thermal_contact]
    #    emissivity_1 = 0.797698     #Emissivity for fuel
    #    emissivity_2 = 0.325        #Emissivity for clad
    slave = 10
    meyer_hardness =  3.4e8
    roughness_clad = 0.5e-6
    jump_distance_model = KENNARD
    roughness_coef = 3.2
    order = SECOND
    plenum_pressure = plenum_pressure
    quadrature = true
    master = 5
    roughness_fuel = 2e-6
    variable = temp
    contact_pressure = contact_pressure
    initial_gas_fractions = '1 0 0 0 0 0 0 0 0 0'
    type = GapHeatTransferLWR
    normal_smoothing_distance = 0.1
  [../]
[]
[BCs]
  # For RELAP-7
  # pin pellets and clad along axis of symmetry (y)
  # pin clad bottom in the axial direction (y)
  # pin fuel bottom in the axial direction (y)
  # RELAP-7 Coupled
  # apply coolant pressure on clad outer walls
  [./convective_clad_surface_2phase]
    T_infinity_liquid = temperature_liquid
    kappa_liquid = kappa_liquid
    Hw_liquid = Hw_liquid
    T_infinity_vapor = temperature_vapor
    Hw_vapor = Hw_vapor
    variable = temp
    kappa_vapor = kappa_vapor
    boundary = '2'
    type = CoupledConvectiveFluxTwoPhase
  [../]
  [./no_x_all]
    variable = disp_x
    boundary = 12
    type = DirichletBC
    value = 0.0
  [../]
  [./no_y_clad_bottom]
    variable = disp_y
    boundary = '1'
    type = DirichletBC
    value = 0.0
  [../]
  [./no_y_fuel_bottom]
    variable = disp_y
    boundary = 1020
    type = DirichletBC
    value = 0.0
  [../]
  [./pressure_bc_x]
    variable = disp_x
    pressure = pressure_mix
    component = 0
    type = CoupledPressureBC
    boundary = '1 2 3'
  [../]
  [./pressure_bc_y]
    variable = disp_y
    pressure = pressure_mix
    component = 1
    type = CoupledPressureBC
    boundary = '1 2 3'
  [../]
  [./PlenumPressure]
    #  apply plenum pressure on clad inner walls and pellet surfaces
    [./plenumPressure]
        #      material_input = fis_gas_released          # coupling to post processor to get fission gas added
        startup_time = 0.0
        initial_temperature = 293.15
        volume =  gas_volume
        initial_pressure = 2.0e6
        R = 8.3143
        temperature =  ave_temp_interior
        output =  plenum_pressure
        boundary = 9
        output_initial_moles =  initial_moles
    [../]
  [../]
[]
[Materials]
  # Model for Zircaloy phase transition
  [./fuel_density]
    type = Density
    disp_r = disp_x
    block = '3'
    disp_z = disp_y
  [../]
  [./fuel_thermal]
    # temperature and burnup dependent thermal properties of UO2 (bison kernel)
    temp = temp
    cp_scalef = 0.977605823121
    thcond_scalef = 1.01376716192
    block = '3'
    model = 4
    type = ThermalFuel
    burnup = burnup
  [../]
  [./fuel_solid]
    #    thermal_expansion = 15e-6
    stress_free_temperature = 297
    large_strain = true
    temp = temp
    poissons_ratio = .345
    thermal_expansion_function =  Fuel_CTE_function
    formulation = AxisymmetricRZ
    youngs_modulus = 2.e11
    type = Elastic
    block = '3'
  [../]
  [./density_clad]
    type = Density
    block = '1'
    density = 6551.0
  [../]
  [./clad_thermal]
    thcond_scalef = 1
    type = ThermalZry
    temp = temp
    block = '1'
  [../]
  [./clad_solid_mechanics]
    stress_free_temperature = 297
    large_strain = true
    temp = temp
    constitutive_model = combined
    poissons_ratio = 0.3
    thermal_expansion = 5.0e-6
    formulation = AxisymmetricRZ
    youngs_modulus = 7.5e10
    type = SolidModel
    block = '1'
    youngs_modulus_function =  YM_function
  [../]
  [./plasticity]
    stress_free_temperature = 297
    temp = temp
    yield_stress_function =  YS_function
    max_its = 100
    thermal_expansion_function = Clad_CTE_function
    hardening_constant = 2.5e9
    absolute_tolerance = 1e-7
    type = IsotropicPlasticity
    block = '1'
  [../]
  [./combined]
    submodels = 'creep plasticity'
    temp = temp
    relative_tolerance = 1e-3
    absolute_tolerance = 1e-3
    type = CombinedCreepPlasticity
    max_its = 2000
    block = '1'
  [../]
  [./creep]
    fast_neutron_fluence = fast_neutron_fluence
    model_thermal_creep_loca = true
    temp = temp
    model_thermal_creep = false
    fast_neutron_flux = fast_neutron_flux
    type = CreepZryModel
    relative_tolerance = 1e-3
    block = '1'
  [../]
  [./phase]
    numerical_method = 2
    type = ZrPhase
    temperature = temp
    block = '1'
  [../]
[]
[Dampers]
  [./limitT]
    variable = temp
    type = MaxIncrement
    max_increment = 30.0
  [../]
[]
[Executioner]
  # PETSC options
  #Preconditioned JFNK (default)
  # Second Order Element Model Options
  # First Order Element Model Options
  #  petsc_options_iname = '-pc_type -pc_factor_mat_solver_package'
  #  petsc_options_value = 'lu       mumps'
  # controls for linear iterations
  # controls for nonlinear iterations
  # time control
  # direct control of time steps vs time (optional)
  #  [./TimeStepper]
  #    type = IterationAdaptiveDT
  #    dt = 0.1
  #    force_step_every_function_point = true
  #    timestep_limiting_function = power_history
  #    max_function_change = 1e20
  #    optimal_iterations = 10
  #    iteration_window = 4
  #    linear_iteration_ratio = 1000
  #    growth_factor = 1.2
  #  [../]
  nl_abs_tol = 1e-10
  petsc_options_value = ' lu       superlu_dist                  51                              0.5                  0.9                  1                  2                   2                    0.1'
  nl_max_its = 25
  type = Transient
  start_time = 0
  dtmax = 1e6
  line_search = 'none'
  l_tol = 1e-4
  nl_rel_tol = 1e-5
  solve_type = 'PJFNK'
  petsc_options = '-snes_ksp_ew'
  dtmin = 1e-6
  petsc_options_iname = '-pc_type -pc_factor_mat_solver_package -ksp_gmres_restart -snes_ksp_ew_rtol0 -snes_ksp_ew_rtolmax -snes_ksp_ew_gamma -snes_ksp_ew_alpha -snes_ksp_ew_alpha2 -snes_ksp_ew_threshold'
  picard_max_its = 2
  l_max_its = 50
  end_time = 501
  [./TimeStepper]
    time_t = '0  100  200  200.02 201.0  501'
    time_dt = '2  10   10   0.02   0.02   20'
    type = FunctionDT
  [../]
  [./Quadrature]
    order = FIFTH
    side_order = SEVENTH
  [../]
[]
[Postprocessors]
  #  [./input_rod_power]
  #    type = FunctionValuePostprocessor
  #    function = power_history
  #  [../]
  [./ave_temp_interior]
    # average temperature of the cladding interior and all pellet exteriors
    variable = temp
    execute_on = linear
    boundary = 9
    type = SideAverageValue
  [../]
  [./gas_volume]
    # gas volume
    execute_on = linear
    boundary = 9
    type = InternalVolume
  [../]
  [./_dt]
    # time step
    type = TimestepSize
  [../]
  [./nonlinear_its]
    # number of nonlinear iterations at each timestep
    type = NumNonlinearIterations
  [../]
  [./max_fuel_centerline_temp]
    variable = temp
    boundary = 12
    type = NodalMaxValue
  [../]
  [./max_clad_outer_surface_temp]
    variable = temp
    boundary = 2
    type = NodalMaxValue
  [../]
  [./clad_inner_vol]
    # volume inside of cladding
    #outputs = exodus
    boundary = 7
    type = InternalVolume
  [../]
  [./pellet_volume]
    # fuel pellet total volume
    boundary = 8
    type = InternalVolume
  [../]
  [./max_alpha_vapor]
    variable = kappa_vapor
    type = NodalMaxValue
  [../]
  [./rod_total_power]
    variable = temp
    burnup_function = burnup
    type = ElementIntegralPower
    block = 3
  [../]
  [./rod_input_power]
    function = power_history
    scale_factor =  4.2672
    type = FunctionValuePostprocessor
  [../]
[]
[Outputs]
  #  [./csv]
  #    type = CSV
  #    execute_vector_postprocessors_on = timestep_end
  #    file_base = uo2_ria
  #  [../]
  #  [./cp]
  #    type = Checkpoint
  #    num_files = 1
  #  [../]
  color = false
  file_base = out~bison_uo2_zry_ria
  interval = 1
  exodus = true
  csv = true
  [./console]
    output_linear = true
    perf_log = true
    type = Console
    max_rows = 25
  [../]
  [./debug]
    type = VariableResidualNormsDebugOutput
  [../]
[]
[Debug]
  show_material_props = true
  show_var_residual_norms = true
  show_var_residual = 'temp'
  show_actions =  true
[]
[MultiApps]
  [./relap]
    max_procs_per_app = 1
    app_type = RELAP7App
    positions = '0 0 0'
    sub_cycling = true
    max_failures = 500
    output_sub_cycles = true
    execute_on = timestep_end
    input_files = relap_uo2_zry_ria.i
    type = TransientMultiApp
    detect_steady_state = true
  [../]
[]
[Transfers]
  # to RELAP-7
  # from RELAP-7
  [./clad_surface_temp_to_relap]
    #    type = MultiAppInterpolationTransfer
    #    fixed_meshes = true
    source_variable = temp
    direction = to_multiapp
    source_boundary = 2
    displaced_target_mesh = true
    variable = Tw
    type = MultiAppNearestNodeTransfer
    multi_app = relap
  [../]
  [./coolant_vapor_to_clad_temp]
    #    type = MultiAppInterpolationTransfer
    #    fixed_meshes = true
    source_variable = temperature_vapor
    direction = from_multiapp
    target_boundary = 2
    variable = temperature_vapor
    displaced_source_mesh = true
    type = MultiAppNearestNodeTransfer
    multi_app = relap
  [../]
  [./coolant_liquid_to_clad_temp]
    #    type = MultiAppInterpolationTransfer
    #    fixed_meshes = true
    source_variable = temperature_liquid
    direction = from_multiapp
    target_boundary = 2
    variable = temperature_liquid
    displaced_source_mesh = true
    type = MultiAppNearestNodeTransfer
    multi_app = relap
  [../]
  [./volume_fraction_vapor_to_clad_temp]
    #    type = MultiAppInterpolationTransfer
    #    fixed_meshes = true
    source_variable = alpha_vapor
    direction = from_multiapp
    target_boundary = 2
    variable = kappa_vapor
    displaced_source_mesh = true
    type = MultiAppNearestNodeTransfer
    multi_app = relap
  [../]
  [./volume_fraction_liquid_to_clad_temp]
    #    type = MultiAppInterpolationTransfer
    #    fixed_meshes = true
    source_variable = alpha_liquid
    direction = from_multiapp
    target_boundary = 2
    variable = kappa_liquid
    displaced_source_mesh = true
    type = MultiAppNearestNodeTransfer
    multi_app = relap
  [../]
  [./Hw_vapor_from_multiapp]
    #    type = MultiAppInterpolationTransfer
    #    fixed_meshes = true
    source_variable = Hw_vapor
    direction = from_multiapp
    target_boundary = 2
    variable = Hw_vapor
    displaced_source_mesh = true
    type = MultiAppNearestNodeTransfer
    multi_app = relap
  [../]
  [./Hw_liquid_from_multiapp]
    #    type = MultiAppInterpolationTransfer
    #    fixed_meshes = true
    source_variable = Hw_liquid
    direction = from_multiapp
    target_boundary = 2
    variable = Hw_liquid
    displaced_source_mesh = true
    type = MultiAppNearestNodeTransfer
    multi_app = relap
  [../]
  [./pressure_mix_from_multiapp]
    #    type = MultiAppInterpolationTransfer
    #    fixed_meshes = true
    source_variable = pressure_mix
    direction = from_multiapp
    target_boundary = 2
    variable = pressure_mix
    displaced_source_mesh = true
    type = MultiAppNearestNodeTransfer
    multi_app = relap
  [../]
[]
