#this input file is used in the rattlesnake regression tests
#we use this input file to test the following capabilities of raven
#1. the moose based app interface can work with rattlesnake
#2. the moose based app interface can handle input files with key values defined in multiple lines (see issue #500)
[Mesh]
  type = GeneratedBIDMesh
  dim = 2
  xmin = 0
  xmax = 80
  ymin = 0
  ymax = 80
  elem_type = QUAD9
  nx = 10
  ny = 10
  subdomain='1 1 1 2 2 2 2 1 1 1
             1 1 1 2 2 2 2 1 1 1
             1 1 1 2 2 2 2 1 1 1
             2 2 2 3 3 3 3 1 1 1
             2 2 2 3 3 3 3 1 1 1
             2 2 2 3 3 3 3 1 1 1
             2 2 2 3 3 3 3 1 1 1
             1 1 1 1 1 1 1 1 1 1
             1 1 1 1 1 1 1 1 1 1
             1 1 1 1 1 1 1 1 1 1'
  uniform_refine = 0
[]

[TransportSystems]
  particle = neutron
  equation_type = eigenvalue

  G = 2

  DirichletBoundary = '1 2'
  ReflectingBoundary = '0 3'

  [./diff]
    scheme = CFEM-Diffusion
    family = LAGRANGE
    order = FIRST
    n_delay_groups = 1
  [../]
[]

[Materials]
  active = 'seed11 blanket1 seed21'

  [./seed11]
    type = MixedNeutronicsMaterial
    block = 2
    library_file = 'xs.xml'
    library_name = 'twigl'
    material_id = 1
    grid_names = temperature
    grid = '1'
    isotopes = 'pseudo-seed1'
    densities = '1.0'
  [../]

  [./blanket1]
    type = MixedNeutronicsMaterial
    block = 1
    library_file = 'xs.xml'
    library_name = 'twigl'
    material_id = 1
    grid_names = temperature
    grid = '1'
    isotopes = 'pseudo-seed2'
    densities = '1.0'
  [../]

  [./seed21]
    type = MixedNeutronicsMaterial
    block = 3
    library_file = 'xs.xml'
    library_name = 'twigl'
    material_id = 1
    grid_names = temperature
    grid = '1'
    isotopes = 'pseudo-seed1-dup'
    densities = '1.0'
  [../]

  [./seed2]
    type = FunctionNeutronicsMaterial
    block = 3
    diffusion_coef = '1.4 0.4'
    sigma_r = '0.02 step_removal'
    sigma_t = '0.23809523809523809523809523809524  0.83333333333333333333333333333333'
    L = 0
    sigma_s = '0.21809523809523809523809523809524 0.0
               0.010 step_scat'
    fissile = true
    nu_sigma_f = '0.007 0.20'
    chi = '1.0 0.0'
    neutron_speed = '1e7 2e5'
    delay_fraction = '7.5e-3'
    decay_constant = '8e-2'
    delay_spectrum = '1.0 0.0'
  [../]
[]

[Executioner]
  type = NonlinearEigen

  #Preconditioned JFNK (default)
  solve_type = 'PJFNK'
  petsc_options_iname = '-pc_type -pc_hypre_type -ksp_gmres_restart '
  petsc_options_value = 'hypre boomeramg 100'

  free_power_iterations = 5
  source_abs_tol = 1e-9
  output_after_power_iterations = false
  output_before_normalization = false
[]

[Postprocessors]
  [./power]
    type = ElementIntegralVariablePostprocessor
    variable = fission_source
  [../]
[]

[Outputs]
  execute_on = 'timestep_end'
  file_base = out1
  exodus = true
[]
