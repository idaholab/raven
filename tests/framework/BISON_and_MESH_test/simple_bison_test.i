[Mesh]
  file = coarse_input.e
[]


[Variables]
  [./temp]
    order = FIRST
    family = LAGRANGE
    initial_condition = 300
  [../]
[]


[AuxVariables]
  [./fission_rate]
    order = FIRST
    family = LAGRANGE
  [../]
[]


[Functions]
  [./axial_power_ramp]
    type = PiecewiseLinear
    x = '0 1e4'
    y = '0 1'
  [../]
[]

[Kernels]
  [./heat]
    type = HeatConduction
    variable = temp
  [../]
[]

[AuxKernels]
  [./fissionrate]
    type = FissionRateAxialAux
    axis = 1
    block = 3
    variable = fission_rate
    value = 1.183e19         # corrected average power to 200 W/cm
    fuel_bottom = -2.5
    fuel_top = 2.5
    function = axial_power_ramp
  [../]
[]

[BCs]
  [./clad_outer_temp]
    type = DirichletBC
    variable = temp
    boundary = 7
    value = 1373.15
  [../]
[]


[Materials]
  [./fuel_thermal]
    type = HeatConductionMaterial
    block = 3
    temp = temp
    thermal_conductivity = 5.0
    specific_heat = 50
  [../]

  [./clad_thermal]
    type = HeatConductionMaterial
    block = 1
    temp = temp
    thermal_conductivity = 16.0
    specific_heat = 330.0
  [../]
[]

[Executioner]
   type = Transient

  #Preconditioned JFNK (default)
  solve_type = 'PJFNK'


   petsc_options = '-snes_ksp_ew '
   petsc_options_iname = '-ksp_gmres_restart -pc_type -pc_hypre_type'
   petsc_options_value = '70 hypre boomeramg'
   l_max_its = 60
   nl_rel_tol = 1e-8
   nl_abs_tol = 1e-8
   l_tol = 1e-5
   start_time = 0.0
   end_time   = 1e+4
   dt = 1e+3
   num_steps = 10
[]

[Postprocessors]
  [./ave_temp_interior]
    type = SideAverageValue
    boundary = 9
    variable = temp
    execute_on = linear
  [../]

  [./flux_from_clad]
    type = SideFluxIntegral
    variable = temp
    boundary = 5
    diffusivity = thermal_conductivity
  [../]

  [./centerline_temp]
    type = NodalVariableValue
    nodeid = 159
    variable = temp
  [../]

  [./clad_outer_temp]
    type = NodalVariableValue
    nodeid = 76
    variable = temp
  [../]
[]

[Outputs]
  file_base = out
  exodus = true
[]
