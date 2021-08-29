[Mesh]
  type = GeneratedMesh
  dim = 1
  xmin = 0
  xmax = 2
  nx = 10
[]
[Variables]
  [./dummy]
  [../]
[]
[Kernels]
  [./dummy_u]
    type = TimeDerivative
    variable = dummy
  [../]
[]
[AuxVariables]
  [./the_linear_combo]
  [../]
[]
[AuxKernels]
  [./the_linear_combo]
    type = FunctionAux
    variable = the_linear_combo
    function = the_linear_combo
  [../]
[]
[Functions]
  [./xtimes]
    type = ParsedFunction
    value = 1.1*x
  [../]
  [./twoxplus1]
    type = ParsedFunction
    value = 2*x+1
  [../]
  [./xsquared]
    type = ParsedFunction
    value = (x-2)*x
  [../]
  [./tover2]
    type = ParsedFunction
    value = 0.5*t
  [../]
  [./the_linear_combo]
    type = LinearCombinationFunction
    functions = "xtimes twoxplus1 xsquared tover2"
    w = "4.8 -1.2 0.6699999999999999 3"
  [../]
  [./should_be_answer]
    type = ParsedFunction
    value = 3*1.1*x-1.2*(2*x+1)+0.4*(x-2)*x+3*0.5*t
  [../]
[]
[Postprocessors]
  [./L2_out]
    type = NodalL2Error
    function = should_be_answer
    variable = the_linear_combo
  [../]
[]
[Executioner]
  type = Transient
  dt = 0.5
  end_time = 1
[]
[Outputs]
  execute_on = timestep_end
  file_base = out~lcf1
  hide = dummy
  exodus = false
  csv = true
[]
