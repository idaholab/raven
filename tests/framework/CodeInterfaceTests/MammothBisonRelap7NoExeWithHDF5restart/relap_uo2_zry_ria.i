#
# Flow through one core channel
#

# Transfer into Tw

[GlobalParams]
  stabilization = evm2

  initial_p_liquid = 15.5e6 # Pa
  initial_p_vapor  = 15.5e6 # Pa
  initial_v_liquid = 0               # 3.71539985296701 # m/s
  initial_v_vapor  = 0               # 3.71539985296701 # m/s
  initial_T_liquid = 293.15  # K
  initial_T_vapor  = 619     # K
  initial_volume_fraction_vapor = 0.01

  # 7 eqn global parameters
  phase_interaction = true
  pressure_relaxation = true
  velocity_relaxation = true
  interface_transfer = true
  wall_mass_transfer = true

  specific_interfacial_area_max_value = 10
  explicit_acoustic_impedance = true
  explicit_alpha_gradient = true
  heat_flux_partitioning_model = linear

  # Scaling factor for "primary variables" of the 7-equation model
  scaling_factor_2phase = '1e0
                           1e0 1e0 1e-4
                           1e0 1e0 1e-4'
  scaling_factor_temperature = 1.e-3 # TODO Figure out what this does!

  gravity = '0.0 -9.81 0.0'
[]

[Stabilizations]
  [./evm1]
    type = EntropyViscosity
    use_first_order_liquid = true
    use_first_order_vapor = true
    use_first_order_vf = true
  [../]
  [./evm2]
    type = EntropyViscosity
    use_first_order_liquid = false
    use_first_order_vapor = false
    use_first_order_vf = false
    use_jump_liquid = true
    use_jump_vapor = true
    use_jump_vf = true
    Cjump_liquid = 10
    Cjump_vapor = 10
    Cjump_vf = 10
  [../]
[]

[FluidProperties]
  [./eos]
    type = IAPWS957EqnFluidProperties
  [../]
[]

[Functions]
  [./Hw_liq_fn]
    type = PiecewiseLinear
#    data_file = Hw_liq_history.csv
    format = columns
    scale_factor = 1
    xy_data = '0       735.9
               100.0   30850
               200.06  30850
               200.22  102000
               200.5   5000
               201.5   4000
               205.2   6200
               215.0   8200
               242.0   14000
               300.0   30850
               501.0   30850'
  [../]

  [./Hw_vap_fn]
    type = PiecewiseConstant
#    data_file = Hw_vap_history.csv
    format = columns
    scale_factor = 1
    xy_data = '0       0
               200.06  2
               200.6   23
               202.0   2
               300.0   0'
  [../]
[]

[Components]
  [./core]
    type = Pipe
    # geometry
    position = '1 0.00224 0' # Starting position of the pipe [m]
    orientation = '0 1 0' # Orientation of pipes path
    heat_transfer_geom = VERTICAL_CYLINDER_BUNDLE_W_CROSS_FLOW
    length = 4.43063 # Fuel rod length [m]
    n_elems = 500 # ~ Match number of "regularly spaced" elements in BISON mesh

    # All of these parameters can be calculated using the core-channel-cylinder.py script
    A   = 8.7877815753E-05    # Cross Sectional Area [m^2]
    Dh  = 1.1777843171E-02    # Hydraulic diameter [m]
    Phf = 2.9845130209E-02    # Heated parameter

    f = 0.01                  # Wall friction factor
    f_interface = 0           # Interface friction factor (needs to be zero)
    Hw_liquid = Hw_liq_fn  # Wall heat transfer coefficient [W /m^2-K]
    Hw_vapor = Hw_vap_fn

    Tw_transferred = true
    #Tw = Tw_fn

    fp = eos
  [../]

  [./inlet]
    type = Inlet
    input = 'core(in)'
    rho_vapor  =  100.0270059202      # Water vapor density [kg/m^3]
    volume_fraction_vapor =  0.01     # Fraction of volume that is water vapor
    # Set by control logic
    rho_liquid = 0          # Liquid water density [kg/m^3]
    u_liquid   = 0          # Liquid water inlet velocity [m/s]
    u_vapor    = 0          # Water vapor inlet velocity [m/s]
  [../]

  [./outlet]
    type = Outlet
    input = 'core(out)'
    p_liquid = 15.5e6   # Same as initial_p_liquid unless pressure drop or set in control logic
    p_vapor = 15.5e6    # Same as initial_p_vapor unless pressure drop or set in control logic
  [../]
[]

[AuxVariables]
  [./pressure_mix]
    family = LAGRANGE
    order = FIRST
  [../]

  [./density_mix]
    family = LAGRANGE
    order = FIRST
  [../]

  [./temp_sat]
    family = LAGRANGE
    order = FIRST
  [../]
[]

[AuxKernels]
  [./pm_aux]
    type = MixtureQuantityAux
    variable = pressure_mix
    a = 'alpha_liquid alpha_vapor'
    b = 'pressure_liquid pressure_vapor'
    execute_on = 'timestep_end'
  [../]

  [./density_aux]
    type = MixtureQuantityAux
    variable = density_mix
    a = 'alpha_liquid alpha_vapor'
    b = 'density_liquid density_vapor'
    execute_on = 'timestep_end'
  [../]

  [./sat_temp]
    type = TemperatureSaturationAux
    variable = temp_sat
    pressure = pressure_liquid
    fp = eos
  [../]
[]

[Controlled]
  [./inlet_rho_liquid]
    component_name = inlet
    property_name = rho_liquid
    data_type = double
  [../]
  [./inlet_u_liquid]
    component_name = inlet
    property_name = u_liquid
    data_type = double
  [../]
  [./inlet_u_vapor]
    component_name = inlet
    property_name = u_vapor
    data_type = double
  [../]
[]

[Preconditioning]
  [./SMP_PJFNK]
    type = SMP
    full = true
    petsc_options = '-snes_converged_reason'
    solve_type = 'PJFNK'
    line_search = basic
  [../]
[]

[Executioner]
  type = ControlLogicExecutioner
  control_logic_file = 'control.py'
  legacy = false

#  type = Transient # For cases when no control logic is required

  # controls for linear iterations
  l_max_its = 50
  l_tol = 1e-2

  # Controls for nonlinear iterations
  nl_max_its = 15
  nl_rel_tol = 1e-6
  nl_abs_tol = 1e-5

  # time control
  # If relap does not converge and multiapps moves to next time step, Turned off to "fix" the multiapps in coupled runs
  scheme = 'bdf2'
  start_time = 0
  end_time = 501
  dtmin = 1.e-8
  # If relap does not converge and multiapps moves to next time step, Turned off to "fix" the multiapps in coupled runs
  [./TimeStepper]
    type = SolutionTimeAdaptiveDT
    dt = 5e-2
  [../]

  [./Quadrature]
    type = TRAP
    order = FIRST
  [../]
[]

[Outputs]
  [./out_displaced]
    type = Exodus
    use_displaced = true
    execute_on = 'initial timestep_end'
    sequence = false
  [../]

  [./console]
    type = Console
    perf_log = true
  [../]
#  checkpoint = true
[]
