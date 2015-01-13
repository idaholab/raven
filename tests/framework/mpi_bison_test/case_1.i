[GlobalParams]
  density = 10800.0          # kg/m^3
  order = SECOND
  family = LAGRANGE
  disp_x = disp_x
  disp_y = disp_y
[]

[Mesh]
  file = case_1.e
  displacements = 'disp_x disp_y'
[]

[Problem]
  coord_type = RZ
[]

[Variables]

  [./disp_x]
  [../]

  [./disp_y]
  [../]

  [./temp]
    initial_condition = 1273.0
  [../]
[]


[AuxVariables]

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

  [./stress_xy]
    order = CONSTANT
    family = MONOMIAL
  [../]

  [./stress_yz]
    order = CONSTANT
    family = MONOMIAL
  [../]

  [./stress_xz]
    order = CONSTANT
    family = MONOMIAL
  [../]

  [./hydrostatic_stress]
    order = CONSTANT
    family = MONOMIAL
  [../]
[]


[SolidMechanics]
  [./solid]
    temp = temp
    disp_r = disp_x
    disp_z = disp_y
  [../]
[]

[Kernels]
  [./heat_ie]
    type = HeatConductionTimeDerivative
    variable = temp
  [../]

  [./heat]
    type = HeatConduction
    variable = temp
  [../]
[]


[AuxKernels]
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

  [./stress_xy]
    type = MaterialTensorAux
    tensor = stress
    variable = stress_xy
    index = 3
    execute_on = timestep_end
  [../]
  [./stress_yz]
    type = MaterialTensorAux
    tensor = stress
    variable = stress_yz
    index = 4
    execute_on = timestep_end
  [../]
  [./stress_xz]
    type = MaterialTensorAux
    tensor = stress
    variable = stress_xz
    index = 5
    execute_on = timestep_end
  [../]
  [./hydrostatic_stress]
    type = MaterialTensorAux
    tensor = stress
    variable = hydrostatic_stress
    quantity = hydrostatic
    execute_on = timestep_end
  [../]
[]

[BCs]

# pin particle along symmetry planes
  [./no_disp_x]
    type = DirichletBC
    variable = disp_x
    boundary = xzero
    value = 0.0
  [../]

  [./no_disp_y]
    type = DirichletBC
    variable = disp_y
    boundary = yzero
    value = 0.0
  [../]

# fix temperature on free surface
  [./freesurf_temp]
    type = DirichletBC
    variable = temp
    boundary = exterior
    value = 1273.0
  [../]

# exterior and internal pressures
  [./exterior_pressure_x]
    type = Pressure
    variable = disp_x
    boundary = exterior
    component = 0
    factor = 0.1e6
  [../]

 [./exterior_pressure_y]
    type = Pressure
    variable = disp_y
    boundary = exterior
    component = 1
    factor = 0.1e6
  [../]

  [./interior_pressure_x]
    type = Pressure
    variable = disp_x
    boundary = PyCGapBndry
    component = 0
    factor = 25e6
  [../]

  [./interior_pressure_y]
    type = Pressure
    variable = disp_y
    boundary = PyCGapBndry
    component = 1
    factor = 25e6
  [../]
[]


[Materials]

  [./fuel_thermal]                     # temperature and burnup dependent thermal properties of UO2 (bison kernel)
    type = ThermalUO2
    block = fuel
    temp = temp
#    burnup = burnup
  [../]

 [./fuel_disp]                         # thermal and irradiation creep for UO2 (bison kernel)
    type = Elastic
    block = fuel
    disp_r = disp_x
    disp_z = disp_y
    temp = temp
#    youngs_modulus = 2.e11
    youngs_modulus = 1
    poissons_ratio = .345
    thermal_expansion = 0
  [../]

  [./fuel_den]
    type = Density
    block = fuel
    disp_r = disp_x
    disp_z = disp_y
  [../]


 [./buffer_disp]                         # thermal and irradiation creep for UO2 (bison kernel)
    type = Elastic
    block = buffer
    disp_r = disp_x
    disp_z = disp_y
    temp = temp
#    youngs_modulus = 2.e11
    youngs_modulus = 1
    poissons_ratio = .345
    thermal_expansion = 0
  [../]

  [./buffer_temp]
    type = HeatConductionMaterial
    block = buffer
    thermal_conductivity = 0.5        # J/m-s-K
    specific_heat = 720.0             # J/kg-K
  [../]

  [./buffer_den]
    type = Density
    density = 950                     #kg/m^3
    block = buffer
    disp_r = disp_x
    disp_z = disp_y
  [../]


  [./SiC_disp]
    type = Elastic
    block = SiC
    temp = temp
    youngs_modulus = 3.7e11
    poissons_ratio = 0.13
    thermal_expansion = 0
    disp_r = disp_x
    disp_z = disp_y
  [../]

  [./SiC_temp]
    type = HeatConductionMaterial
    block = SiC
    thermal_conductivity = 13.9          # J/m-s-K
    specific_heat = 620.0                # J/kg-K
  [../]

  [./SiC_den]
    type = Density
    density = 3180.0                     # kg/m^3
    block = SiC
    disp_r = disp_x
    disp_z = disp_y
  [../]
[]

#[Preconditioning]
#  [./SMP]
#    type = SMP
#    full = true
#  [../]
#[]

[Dampers]
  [./temp]
    type = MaxIncrement
    variable = temp
    max_increment = 50
  [../]
[]

[Debug]
    show_var_residual_norms = true
[]

[Executioner]
  type = Transient

  petsc_options_iname = '-ksp_gmres_restart -pc_type -pc_hypre_type -pc_hypre_boomeramg_max_iter'
  petsc_options_value = '201                hypre    boomeramg      4'

  line_search = 'none'


  #Preconditioned JFNK (default)
  solve_type = 'PJFNK'




  nl_rel_tol = 5e-6
  nl_abs_tol = 1e-10
  nl_max_its = 15

  l_tol = 1e-3
  l_max_its = 50

   start_time = 0.0
 #  end_time = 80e6
   end_time = 10
   num_steps = 10000

  dtmax = 5e6
  dtmin = 1
  [./TimeStepper]
    type = IterationAdaptiveDT
    dt = 1
    optimal_iterations = 6
    iteration_window = 0.4
    linear_iteration_ratio = 100
  [../]

  [./Predictor]
    type = SimplePredictor
    scale = 1.0
  [../]

#  [./Quadrature]
#    order = THIRD
#  [../]
[]


[Postprocessors]

  [./dt]
    type = TimestepSize
  [../]
[]

[Output]
  linear_residuals = true
   interval = 1
   output_initial = true
   exodus = true
   postprocessor_csv = true
   perf_log = true
[]
