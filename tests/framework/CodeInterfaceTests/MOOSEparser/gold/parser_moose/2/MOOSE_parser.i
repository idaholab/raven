[Mesh]
  dim = 1
  dx = '20 20 200.0'
  ix = '5 5 40'
  type = CartesianMesh
  uniform_refine = 1
[]
[Problem]
  coord_type = RSPHERICAL
  kernel_coverage_check = false
[]
[GlobalParams]
  family = MONOMIAL
  order = CONSTANT
[]
[Variables]
  [./temperature]
    family = LAGRANGE
    initial_condition = 300
    order = FIRST
  [../]
[]
[Kernels]
  [./HeatConduction]
    diffusion_coefficient_dT = dthermal_conductivity/dtemperature
    type = HeatConduction
    variable = temperature
  [../]
  [./HeatStorage]
    heat_capacity = cp_rho
    type = HeatCapacityConductionTimeDerivative
    variable = temperature
  [../]
  [./HeatSource]
    type = CoupledForce
    v = heat_source
    variable = temperature
  [../]
[]
[AuxVariables]
  [./rho_1]
  [../]
  [./heat_source]
  [../]
  [./current_power_density]
  [../]
  [./radius]
    family = LAGRANGE
    order = FIRST
  [../]
  [./fuel_temperature]
  [../]
  [./graphite_temperature]
  [../]
  [./thermal_conductivity]
  [../]
  [./nsh]
    initial_condition = 1226.500730082983
  [../]
[]
[AuxKernels]
  [./thermal_conductivity]
    property = thermal_conductivity
    type = MaterialRealAux
    variable = thermal_conductivity
  [../]
  [./radius]
    execute_on = initial
    function = set_r
    type = FunctionAux
    variable = radius
  [../]
  [./rho_1]
    args = 'radius'
    constant_expressions = '20'
    constant_names = 'grain_radius'
    execute_on = initial
    function = 'if(radius < grain_radius, 1, 0)'
    type = ParsedAux
    variable = rho_1
  [../]
  [./heat_source]
    args = 'current_power_density radius'
    constant_expressions = '0.1  20           86000000.0  40     1.0e-6           1'
    constant_names = 'tau   grain_radius total_volume cutoff unit_conversion fr_normalization'
    execute_on = timestep_begin
    function = 'alpha := total_volume * current_power_density * (1 - tau); beta := tau * current_power_density; frf := 2.0e-05 - 2.25e-08 * pow(radius, 2); frg := 5.5e-07 * pow(radius, 2) - 1.5e-05 * radius + 2.0e-04; fr := if(radius < grain_radius, frf, if(radius < cutoff, frg, 0)) / fr_normalization; unit_conversion * (alpha * fr + beta)'
    type = ParsedAux
    variable = heat_source
  [../]
  [./current_power_density]
    execute_on = timestep_begin
    function = local_power_density
    type = FunctionAux
    variable = current_power_density
  [../]
  [./fuel_temperature]
    args = 'temperature rho_1'
    constant_expressions = '38000.0'
    constant_names = 'fuel_volume'
    execute_on = timestep_end
    function = 'rho_1 * temperature / fuel_volume'
    type = ParsedAux
    variable = fuel_temperature
  [../]
  [./graphite_temperature]
    args = 'temperature rho_1'
    constant_expressions = '86000000.0'
    constant_names = 'graphite_volume'
    execute_on = timestep_end
    function = '(1 - rho_1) * temperature / graphite_volume'
    type = ParsedAux
    variable = graphite_temperature
  [../]
[]
[Materials]
  [./thermalconductivity_mat]
    args = 'temperature radius nsh'
    constant_expressions = '1000.0 2.8e9'
    constant_names = 'r      energy'
    derivative_order = 1
    f_name = thermal_conductivity
    function = 'k_fuel := 100 / (6.5 + 25.5 * temperature / r) + 6400 / pow(temperature / r, 2.5) * exp(-10.35 * r / temperature); dpa := 8.6e-18 * pow(radius, 5) -6.51e-16 * pow(radius, 4) + 2.5e-14 * pow(radius, 3) -5.0e-13 * pow(radius, 2) + 4.2e-12 * radius -1.2e-11; dpat := if(radius > 20 & radius < 40, dpa, 0) * nsh * energy; k_graphite := 40.0 * (1.0 / 39.0 + (1.0 - 1.0 / 39.0) * exp(-200 * dpat)); if(radius < 20, k_fuel, k_graphite)'
    type = DerivativeParsedMaterial
  [../]
  [./cp_rho_mat]
    args = 'temperature rho_1'
    constant_expressions = '11.0   260    1.7'
    constant_names = 'rho_fuel M_fuel rho_graphite'
    f_name = cp_rho
    function = 'rT := temperature / 1000.0; cp_f := 50.0 + 90.0 * rT - 82.2 * pow(rT, 2) + 30.5 * pow(rT, 3) - 2.6 * pow(rT, 4) - 0.7 * pow(rT, 4); cp_rho_fuel := 1.0e-6 * rho_fuel / M_fuel * cp_f; cp_rho_graphite := 1e-9 * rho_graphite / (10.0 * pow(temperature, -1.4) + 0.00038 * pow(temperature, 0.029)); rho_1 * cp_rho_fuel + (1 - rho_1) * cp_rho_graphite'
    type = ParsedMaterial
  [../]
  [./density_mat]
    prop_names = 'density'
    prop_values = '1'
    type = GenericConstantMaterial
  [../]
[]
[Functions]
  [./local_power_density]
    type = PiecewiseLinear
    x = '0.0 1.0'
    y = '1  2'
  [../]
  [./set_r]
    type = ParsedFunction
    value = 'x'
  [../]
[]
[Postprocessors]
  [./total_heat_source]
    type = ElementIntegralVariablePostprocessor
    variable = heat_source
  [../]
  [./average_fuel_temperature]
    type = ElementIntegralVariablePostprocessor
    variable = fuel_temperature
  [../]
  [./average_graphite_temperature]
    type = ElementIntegralVariablePostprocessor
    variable = graphite_temperature
  [../]
  [./delta_temp]
    type = DifferencePostprocessor
    value1 = average_fuel_temperature
    value2 = average_graphite_temperature
  [../]
[]
[Executioner]
  end_time = 2.0e-3
  l_max_its = 100
  l_tol = 1.0e-3
  nl_abs_tol = 1e-5
  nl_rel_tol = 1e-8
  petsc_options_iname = '-pc_type -pc_hypre_type -ksp_gmres_restart -pc_hypre_boomeramg_max_iter -pc_hypre_boomeramg_tol'
  petsc_options_value = 'hypre boomeramg 100 20 1.0e-6'
  type = Transient
  [./TimeStepper]
    dt = 1.0e-3
    growth_factor = 1.5
    type = ConstantDT
  [../]
[]
[Outputs]
  csv = true
  exodus = false
  file_base = out~MOOSE_parser
[]
