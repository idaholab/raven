[Mesh]
  dim = 1
  type = CartesianMesh
  uniform_refine = 1
  dx = '20 20 200.0'
  ix = '5 5 40'
[]
[Problem]
  kernel_coverage_check = false
  coord_type = RSPHERICAL
[]
[GlobalParams]
  order = CONSTANT
  family = MONOMIAL
[]
[Variables]
  [./temperature]
    order = FIRST
    family = LAGRANGE
    initial_condition = 300
  [../]
[]
[Kernels]
  [./HeatConduction]
    variable = temperature
    diffusion_coefficient_dT = dthermal_conductivity/dtemperature
    type = HeatConduction
  [../]
  [./HeatStorage]
    variable = temperature
    heat_capacity = cp_rho
    type = HeatCapacityConductionTimeDerivative
  [../]
  [./HeatSource]
    variable = temperature
    type = CoupledForce
    v = heat_source
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
    order = FIRST
    family = LAGRANGE
  [../]
  [./fuel_temperature]
  [../]
  [./graphite_temperature]
  [../]
  [./thermal_conductivity]
  [../]
  [./nsh]
    initial_condition = 10
  [../]
[]
[AuxKernels]
  [./thermal_conductivity]
    variable = thermal_conductivity
    property = thermal_conductivity
    type = MaterialRealAux
  [../]
  [./radius]
    variable = radius
    function = set_r
    type = FunctionAux
    execute_on = initial
  [../]
  [./rho_1]
    constant_expressions = '20'
    function = 'if(radius < grain_radius, 1, 0)'
    args = 'radius'
    constant_names = 'grain_radius'
    variable = rho_1
    execute_on = initial
    type = ParsedAux
  [../]
  [./heat_source]
    constant_expressions = '0.1  20           86000000.0  40     1.0e-6           1'
    function = 'alpha := total_volume * current_power_density * (1 - tau); beta := tau * current_power_density; frf := 2.0e-05 - 2.25e-08 * pow(radius, 2); frg := 5.5e-07 * pow(radius, 2) - 1.5e-05 * radius + 2.0e-04; fr := if(radius < grain_radius, frf, if(radius < cutoff, frg, 0)) / fr_normalization; unit_conversion * (alpha * fr + beta)'
    args = 'current_power_density radius'
    constant_names = 'tau   grain_radius total_volume cutoff unit_conversion fr_normalization'
    variable = heat_source
    execute_on = timestep_begin
    type = ParsedAux
  [../]
  [./current_power_density]
    variable = current_power_density
    function = local_power_density
    type = FunctionAux
    execute_on = timestep_begin
  [../]
  [./fuel_temperature]
    constant_expressions = '38000.0'
    function = 'rho_1 * temperature / fuel_volume'
    args = 'temperature rho_1'
    constant_names = 'fuel_volume'
    variable = fuel_temperature
    execute_on = timestep_end
    type = ParsedAux
  [../]
  [./graphite_temperature]
    constant_expressions = '86000000.0'
    function = '(1 - rho_1) * temperature / graphite_volume'
    args = 'temperature rho_1'
    constant_names = 'graphite_volume'
    variable = graphite_temperature
    execute_on = timestep_end
    type = ParsedAux
  [../]
[]
[Materials]
  [./thermalconductivity_mat]
    constant_expressions = '1000.0 2.8e9'
    function = 'k_fuel := 100 / (6.5 + 25.5 * temperature / r) + 6400 / pow(temperature / r, 2.5) * exp(-10.35 * r / temperature); dpa := 8.6e-18 * pow(radius, 5) -6.51e-16 * pow(radius, 4) + 2.5e-14 * pow(radius, 3) -5.0e-13 * pow(radius, 2) + 4.2e-12 * radius -1.2e-11; dpat := if(radius > 20 & radius < 40, dpa, 0) * nsh * energy; k_graphite := 40.0 * (1.0 / 39.0 + (1.0 - 1.0 / 39.0) * exp(-200 * dpat)); if(radius < 20, k_fuel, k_graphite)'
    f_name = thermal_conductivity
    args = 'temperature radius nsh'
    derivative_order = 1
    constant_names = 'r      energy'
    type = DerivativeParsedMaterial
  [../]
  [./cp_rho_mat]
    constant_expressions = '11.0   260    1.7'
    function = 'rT := temperature / 1000.0; cp_f := 50.0 + 90.0 * rT - 82.2 * pow(rT, 2) + 30.5 * pow(rT, 3) - 2.6 * pow(rT, 4) - 0.7 * pow(rT, 4); cp_rho_fuel := 1.0e-6 * rho_fuel / M_fuel * cp_f; cp_rho_graphite := 1e-9 * rho_graphite / (10.0 * pow(temperature, -1.4) + 0.00038 * pow(temperature, 0.029)); rho_1 * cp_rho_fuel + (1 - rho_1) * cp_rho_graphite'
    f_name = cp_rho
    args = 'temperature rho_1'
    constant_names = 'rho_fuel M_fuel rho_graphite'
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
    x = '0.0 1.0'
    y = '1  2'
    type = PiecewiseLinear
  [../]
  [./set_r]
    type = ParsedFunction
    value = 'x'
  [../]
[]
[Postprocessors]
  [./total_heat_source]
    variable = heat_source
    type = ElementIntegralVariablePostprocessor
  [../]
  [./average_fuel_temperature]
    variable = fuel_temperature
    type = ElementIntegralVariablePostprocessor
  [../]
  [./average_graphite_temperature]
    variable = graphite_temperature
    type = ElementIntegralVariablePostprocessor
  [../]
  [./delta_temp]
    type = DifferencePostprocessor
    value2 = average_graphite_temperature
    value1 = average_fuel_temperature
  [../]
[]
[Executioner]
  nl_abs_tol = 1e-5
  petsc_options_value = 'hypre boomeramg 100 20 1.0e-6'
  petsc_options_iname = '-pc_type -pc_hypre_type -ksp_gmres_restart -pc_hypre_boomeramg_max_iter -pc_hypre_boomeramg_tol'
  l_tol = 1.0e-3
  nl_rel_tol = 1e-8
  end_time = 2.0e-3
  l_max_its = 100
  type = Transient
  [./TimeStepper]
    growth_factor = 1.5
    dt = 1.0e-3
    type = ConstantDT
  [../]
[]
[Outputs]
  csv = true
  exodus = false
[]

