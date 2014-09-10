#
# 1-D Gap Perfect Heat Transfer
#
# This test exercises that 1-D gap heat transfer for a helium-filled gas.
#
# The mesh consists of two element blocks containing one element each.  Each
#   element is a unit cube.  They sit next to one another with a unit between
#   them.
#
# The temperature of the far left boundary is ramped from 100 to 200 over one
#   second and then held fixed.  The temperature of the far right boundary
#   follows due to the perfect heat transfer.
#

[Mesh]
  file = gap_perfect_transfer_test.e
  displacements = 'displ_x displ_y displ_z'
[]

[Functions]
  [./temp]
    type = PiecewiseLinear
    x = '0   1   2'
    y = '100 200 200'
  [../]
[]

[Variables]
  [./displ_x]
    order = FIRST
    family = LAGRANGE
  [../]

  [./displ_y]
    order = FIRST
    family = LAGRANGE
  [../]

  [./displ_z]
    order = FIRST
    family = LAGRANGE
  [../]

  [./temp]
    order = FIRST
    family = LAGRANGE
    initial_condition = 100
  [../]
[]

[SolidMechanics]
  [./solid]
    disp_x = displ_x
    disp_y = displ_y
    disp_z = displ_z
  [../]
[]

[Kernels]
  [./heat]
    type = HeatConduction
    variable = temp
  [../]
[]

[BCs]
  [./fixed_x]
    type = DirichletBC
    boundary = '1 2 3 4'
    variable = displ_x
    value = 0
  [../]

  [./fixed_y]
    type = DirichletBC
    boundary = '1 2 3 4'
    variable = displ_y
    value = 0
  [../]

  [./fixed_z]
    type = DirichletBC
    boundary = '1 2 3 4'
    variable = displ_z
    value = 0
  [../]

  [./temp_far_left]
    type = FunctionDirichletBC
    boundary = 1
    variable = temp
    function = temp
  [../]
[]

[ThermalContact]
  [./thermal_contact_1]
    type = GapPerfectConductance

    variable = temp
    master = 3
    slave = 2
  [../]
  [./thermal_contact_2]
    type = GapHeatTransferLWR

    variable = temp
    master = 2
    slave = 3
  [../]
[]

[Materials]

  [./dummy]
    type = Elastic
    block = '1 2'

    disp_x = displ_x
    disp_y = displ_y
    disp_z = displ_z

    youngs_modulus = 1e6
    poissons_ratio = .3

    temp = temp
    thermal_expansion = 0
  [../]

  [./heat1]
    type = HeatConductionMaterial
    block = 1

    specific_heat = 1.0
    thermal_conductivity = 1.0
  [../]

  [./heat2]
    type = HeatConductionMaterial
    block = 2

    specific_heat = 1.0
    thermal_conductivity = 10.0
  [../]
  [./density]
    type = Density
    block = '1 2'
    density = 1.0
  [../]
[]

[Executioner]
  type = Transient

  #Preconditioned JFNK (default)
  solve_type = 'PJFNK'



  petsc_options_iname = '-ksp_gmres_restart -pc_type -pc_hypre_type -pc_hypre_boomeramg_max_iter'
  petsc_options_value = '201                hypre    boomeramg      4'

  line_search = 'none'


  nl_abs_tol = 1e-8
  nl_rel_tol = 1e-14

  l_tol = 1e-3
  l_max_its = 100

  start_time = 0.0
  dt = 1e-1
  end_time = 2.0
  num_steps = 5000
[]

[Postprocessors]

  [./aveTempLeft]
    type = SideAverageValue
    boundary = 1
    variable = temp
  [../]
  [./aveTempRight]
    type = SideAverageValue
    boundary = 4
    variable = temp
  [../]
[]

[Outputs]
  #linear_residuals = true
  file_base = out
  interval = 1
  output_initial = true
  exodus = true
  #perf_log = true
[]
