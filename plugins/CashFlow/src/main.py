"""
  Execution for CashFlow
"""
import os
import sys
import functools
from collections import defaultdict

import numpy as np
try:
  from CashFlow.src import CashFlows
  # NOTE this import exception is ONLY to allow RAVEN to directly import this extmod.
  # In general, this should not exist, and RAVEN should import CashFlow.CashFlow_ExtMod instead of importing CashFlow_ExtMod directly, implicitly.
except (ImportError, ModuleNotFoundError):
  import CashFlows

raven_path= os.path.abspath(os.path.dirname(__file__)) + '/../../raven/framework'
sys.path.append(raven_path) #'~/projects/raven/framework') # TODO generic RAVEN location

from utils.graphStructure import graphObject
from utils import mathUtils as utils

#=====================
# UTILITIES
#=====================
def read_from_xml(xml):
  """
    reads in cash flow from XML
    @ In, xml, xml.etree.ElementTree.Element, "Economics" node from input
    @ Out, global_settings, CashFlows.GlobalSettings instance, settings for a run (None if none provided)
    @ Out, components, list, CashFlows.Components instances for a run
  """
  # read in XML to global settings, component list
  attr = xml.attrib
  global_settings = None
  components = []
  econ = xml.find('Economics')
  verb = int(econ.attrib.get('verbosity', 100))
  for node in econ:
    if node.tag == 'Global':
      global_settings = CashFlows.GlobalSettings(**attr)
      global_settings.read_input(node)
      global_settings._verbosity = verb
    elif node.tag == 'Component':
      new = CashFlows.Component(**attr)
      new.read_input(node)
      components.append(new)
    else:
      raise IOError('Unrecognized node under <Economics>: {}'.format(node.tag))
  return global_settings, components

def check_run_settings(settings, components):
  """
    Checks that basic settings between global and components are satisfied.
    Errors out if any problems are found.
    @ In, settings, CashFlows.GlobalSettings, global settings
    @ In, components, list, list of CashFlows.Component instances
    @ Out, None
  """
  comp_by_name = dict((c.name, c) for c in components)
  # perform final checks for the global settings and components
  for find, find_cf in settings.get_active_components().items():
    if find not in comp_by_name:
      raise IOError('Requested active component "{}" but not found! Options are: {}'.format(find, list(comp_by_name.keys())))
    # check cash flow is in comp
  # check that StartTime/Repetitions triggers a ProjectTime node
  ## if projecttime is not given, then error if start time/repetitions given (otherwise answer is misleading)
  if settings.get_project_time() is None:
    for comp in components:
      warn = 'CashFlow: <{node}> given for component "{comp}" but no <ProjectTime> in global settings!'
      if comp.get_start_time() != 0:
        raise IOError(warn.format(node='StartTime', comp=comp.name))
      if comp.get_repetitions() != 0:
        raise IOError(warn.format(node='Repetitions', comp=comp.name))
  # check that if NPV_search is an indicator, then mult_target is on at least one cashflow.
  if 'NPV_search' in settings.get_indicators() and sum(comp.count_multtargets() for comp in components) < 1:
    raise IOError('NPV_search in <Indicators> "name" but no cash flows have "mult_target=True"!')

def check_drivers(settings, components, variables, v=100):
  """
    checks if all drivers needed are present in variables
    @ In, settings, CashFlows.GlobalSettings, global settings
    @ In, components, list, list of CashFlows.Component instances
    @ In, variables, dict, variable-value map from RAVEN
    @ In, v, int, verbosity level
    @ Out, ordered, list, list of ordered cashflows to evaluate (in order)
  """
  m = 'check_drivers'
  #active = _get_active_drivers(settings, components)
  active = list(comp for comp in components if comp.name in settings.get_active_components())
  vprint(v, 0, m, '... creating evaluation sequence ...')
  ordered = _create_eval_process(active, variables)
  vprint(v, 0, m, '... evaluation sequence:', ordered)
  return ordered

def _create_eval_process(components, variables):
  """
    Sorts the cashflow evaluation process so sensible evaluation order is used
    @ In, components, list, list of CashFlows.Component instances
    @ In, variables, dict, variable-value map from RAVEN
    @ Out, ordered, list, list of ordered cashflows to evaluate (in order)
  """
  # TODO does this work with float drivers (e.g. already-evaluated drivers)?
  # storage for creating graph sequence
  driver_graph = defaultdict(list)
  driver_graph['EndNode'] = []
  evaluated = [] # for cashflows that have already been evaluated and don't need more treatment
  for comp in components:
    lifetime = comp.get_lifetime()
    # find multiplier variables
    multipliers = comp.get_multipliers()
    for mult in multipliers:
      if mult is None:
        continue
      if mult not in variables.keys():
        raise RuntimeError('CashFlow: multiplier "{}" required for Component "{}" but not found among variables!'.format(mult, comp.name))
    # find order in which to evaluate cash flow components
    for c, cf in enumerate(comp.get_cashflows()):
      # keys for graph are drivers, cash flow names
      driver = cf.get_param('driver')
      # does the driver come from the variable list, or from another cashflow, or is it already evaluated?
      cfn = '{}|{}'.format(comp.name, cf.name)
      found = False
      if driver is None or utils.isAFloatOrInt(driver) or isinstance(driver, np.ndarray):
        found = True
        # TODO assert it's already filled?
        evaluated.append(cfn)
        continue
      elif driver in variables:
        found = True
        # check length of driver
        n = len(np.atleast_1d(variables[driver]))
        if n > 1 and n != lifetime+1:
          raise RuntimeError(('Component "{c}" CashFlow {cf} driver variable "{d}" has "{n}" entries, '+\
                              'but "{c}" has a lifetime of {el}!')
                             .format(c=comp.name,
                                     cf=cf.name,
                                     d=driver,
                                     n=n,
                                     el=lifetime))
      else:
        # driver should be in cash flows if not in variables
        driver_comp, driver_cf = driver.split('|')
        for match_comp in components:
          if match_comp.name == driver_comp:
            # for cross-referencing, component lifetimes have to be the same!
            if match_comp.get_lifetime() != comp.get_lifetime():
              raise RuntimeError(('Lifetimes for Component "{d}" and cross-referenced Component {m} ' +\
                                  'do not match, so no cross-reference possible!')
                                 .format(d=driver_comp, m=match_comp.name))
            found = True # here this means that so far the component was found, not the specific cash flow.
            break
        else:
          found = False
        # if the component was found, check the cash flow is part of the component
        if found:
          if driver_cf not in list(m_cf.name for m_cf in match_comp.get_cashflows()):
            found = False
      if not found:
        raise RuntimeError(('Component "{c}" CashFlow {cf} driver variable "{d}" was not found ' +\
                            'among variables or other cashflows!')
                           .format(c=comp.name,
                                   cf=cf.name,
                                   d=driver))

      # assure each cashflow is in the mix, and has an EndNode to rely on (helps graph construct accurately)
      driver_graph[cfn].append('EndNode')
      # each driver depends on its cashflow
      driver_graph[driver].append(cfn)
  return evaluated + graphObject(driver_graph).createSingleListOfVertices()

def component_life_cashflow(comp, cf, variables, lifetime_cashflows, v=100):
  """
    Calcualtes the annual lifetime-based cashflow for a cashflow of a component
    @ In, comp, CashFlows.Component, component whose cashflow is being analyzed
    @ In, cf, CashFlows.CashFlow, cashflow who is being analyzed
    @ In, variables, dict, RAVEN variables as name: value
    @ In, v, int, verbosity
    @ Out, life_cashflow, np.array, array of cashflow values with length of component life
  """
  m = 'comp_life'
  vprint(v, 1, m, "-"*75)
  print('DEBUGG comp:', comp.name, cf)
  vprint(v, 1, m, 'Computing LIFETIME cash flow for Component "{}" CashFlow "{}" ...'.format(comp.name, cf.name))
  param_text = '... {:^10.10s}: {: 1.9e}'
  # do cashflow
  results = cf.calculate_cashflow(variables, lifetime_cashflows, comp.get_lifetime()+1, v)
  life_cashflow = results['result']

  if v < 1:
    # print out all of the parts of the cashflow calc
    for item, value in results.items():
      if item == 'result':
        continue
      if utils.isAFloatOrInt(value):
        vprint(v, 1, m, param_text.format(item, value))
      else:
        orig = cf.get_param(item)
        if utils.isSingleValued(orig):
          name = orig
        else:
          name = '(from input)'
        vprint(v, 1, m, '... {:^10.10s}: {}'.format(item, name))
        vprint(v, 1, m, '...           mean: {: 1.9e}'.format(value.mean()))
        vprint(v, 1, m, '...           std : {: 1.9e}'.format(value.std()))
        vprint(v, 1, m, '...           min : {: 1.9e}'.format(value.min()))
        vprint(v, 1, m, '...           max : {: 1.9e}'.format(value.max()))
        vprint(v, 1, m, '...           nonz: {:d}'.format(np.count_nonzero(value)))

    yx = max(len(str(len(life_cashflow))),4)
    vprint(v, 0, m, 'LIFETIME cash flow summary by year:')
    vprint(v, 0, m, '    {y:^{yx}.{yx}s}, {a:^10.10s}, {d:^10.10s}, {c:^15.15s}'.format(y='year',
                                                                                        yx=yx,
                                                                                        a='alpha',
                                                                                        d='driver',
                                                                                        c='cashflow'))
    for y, cash in enumerate(life_cashflow):
      if cf.type in ['Capex']:
        vprint(v, 1, m, '    {y:^{yx}d}, {a: 1.3e}, {d: 1.3e}, {c: 1.9e}'.format(y=y,
                                                                                 yx=yx,
                                                                                 a=results['alpha'][y],
                                                                                 d=results['driver'][y],
                                                                                 c=cash))
      elif cf.type == 'Recurring':
        vprint(v, 1, m, '    {y:^{yx}d}, -- N/A -- , -- N/A -- , {c: 1.9e}'.format(y=y,
                                                           yx=yx,
                                                           c=cash))
  return life_cashflow

def get_project_length(settings, components, v=100):
  """
    checks if all drivers needed are present in variables
    @ In, settings, CashFlows.GlobalSettings, global settings
    @ In, components, list, list of CashFlows.Component instances
    @ In, v, int, verbosity level
    @ Out, project_length, int, length of project (explicit or implicit)
  """
  m = 'get_project_length'
  project_length = settings.get_project_time()
  if not project_length:
    vprint(v, 0, m, 'Because project length was not specified, using least common multiple of component lifetimes.')
    lifetimes = list(c.get_lifetime() for c in components)
    project_length = lcmm(*lifetimes) + 1
  return int(project_length)

def project_life_cashflows(settings, components, lifetime_cashflows, project_length, v=100):
  """
    creates all cashflows for life of project, for all components
    @ In, settings, CashFlows.GlobalSettings, global settings
    @ In, components, list, list of CashFlows.Component instances
    @ In, lifetime_cashflows, dict, component: cashflow: np.array of annual economic values
    @ In, project_length, int, project years
    @ In, v, int, verbosity level
    @ Out, project_cashflows, dict, dictionary of project-length cashflows (same structure as lifetime dict)
  """
  m = 'proj_life'
  # apply tax, inflation
  project_cashflows = {} # same keys as lifetime_cashflows
  for comp in components:
    tax = comp.get_tax() if comp.get_tax() is not None else settings.get_tax()
    inflation = comp.get_inflation() if comp.get_inflation() is not None else settings.get_inflation()
    comp_proj_cashflows = project_component_cashflows(comp, tax, inflation, lifetime_cashflows[comp.name], project_length, v=v)
    project_cashflows[comp.name] = comp_proj_cashflows
  return project_cashflows

def project_component_cashflows(comp, tax, inflation, life_cashflows, project_length, v=100):
  """
    does all the cashflows for a SINGLE COMPONENT for the life of the project
    @ In, comp, CashFlows.Component, component to run numbers for
    @ In, tax, float, tax rate for component as decimal
    @ In, inflation, float, inflation rate as decimal
    @ In, life_cashflows, dict, dictionary of component lifetime cash flows
    @ In, project_length, int, project years
    @ In, v, int, verbosity level
    @ Out, cashflows, dict, dictionary of cashflows for this component, taken to project life
  """
  m = 'proj comp'
  vprint(v, 1, m, "-"*75)
  vprint(v, 1, m, 'Computing PROJECT cash flow for Component "{}" ...'.format(comp.name))
  cashflows = {}
  # what is the first project year this component will be in existence?
  comp_start = comp.get_start_time()
  # how long does each build of this component last?
  comp_life = comp.get_lifetime()
  # what is the last project year this component will be in existence?
  ## TODO will this work properly if start time is negative? Initial tests say yes ...
  ## note that we use project_length as the default END of the component's cashflow life, NOT a decomission year!
  comp_end = project_length if comp.get_repetitions() == 0 else comp_start + comp_life * comp.get_repetitions()
  vprint(v, 1, m, ' ... component start: {}'.format(comp_start))
  vprint(v, 1, m, ' ... component end:   {}'.format(comp_end))
  for cf in comp.get_cashflows():
    if cf.is_taxable():
      tax_mult = 1.0 - tax
    else:
      tax_mult = 1.0
    if cf.is_inflated():
      infl_rate = inflation + 1.0
    else:
      infl_rate = 1.0 # TODO nominal inflation rate?
    vprint(v, 1, m, ' ... inflation rate: {}'.format(infl_rate))
    vprint(v, 1, m, ' ... tax rate: {}'.format(tax_mult))
    life_cf = life_cashflows[cf.name]
    single_cashflow = project_single_cashflow(cf, comp_start, comp_end, comp_life, life_cf, tax_mult, infl_rate, project_length, v=v)
    vprint(v, 0, m, 'Project Cashflow for Component "{}" CashFlow "{}":'.format(comp.name, cf.name))
    if v < 1:
      vprint(v, 0, m, 'Year, Time-Adjusted Value')
      for y, val in enumerate(single_cashflow):
        vprint(v, 0, m, '{:4d}: {: 1.9e}'.format(y, val))
    cashflows[cf.name] = single_cashflow
  return cashflows

def project_single_cashflow(cf, start, end, life, life_cf, tax_mult, infl_rate, project_length, v=100):
  """
    does a single cashflow for the life of the project
    @ In, cf, CashFlows.CashFlow, cash flow to extend to full project life
    @ In, start, int, project year in which component begins operating
    @ In, end, int, project year in which component ends operating
    @ In, life, int, lifetime of component
    @ In, life_cf, np.array, cashflow for lifetime of component
    @ In, tax_mult, float, tax rate multiplyer (1 - tax)
    @ In, infl_rate, float, inflation rate multiplier (1 - inflation)
    @ In, project_length, int, total years of analysis
    @ In, v, int, verbosity
    @ Out, proj_cf, np.array, cashflow for project life of component
  """
  m = 'proj c_fl'
  vprint(v, 1, m, "-"*50)
  vprint(v, 1, m, 'Computing PROJECT cash flow for CashFlow "{}" ...'.format(cf.name))
  proj_cf = np.zeros(project_length)
  years = np.arange(project_length) # years in project time, year 0 is first year # TODO just indices, pandas?
  # before the project starts, after it ends are zero; we want the working part
  operating_mask = np.logical_and(years >= start, years <= end)
  operating_years = years[operating_mask]
  start_shift = operating_years - start # y_shift
  # what year realative to production is this component in, for each operating year?
  relative_operation = start_shift % life # yReal
  # handle new builds
  ## three types of new builds:
  ### 1) first ever build (only construction cost)
  ### 2) decomission after last year ever running (assuming said decomission is inside the operational years)
  ### 3) years with both a decomissioning and a construction
  ## this is all years in which construction will occur (covers 1 and half of 3)
  new_build_mask = [a[relative_operation==0] for a in np.where(operating_mask)]
  # NOTE make the decomission_mask BEFORE removing the last-year-rebuild, if present.
  ## This lets us do smoother numpy operations.
  decomission_mask = [new_build_mask[0][1:]]
  # if the last year is a rebuild year, don't rebuild, as it won't be operated.
  if new_build_mask[0][-1] == years[-1]:
    new_build_mask[0] = new_build_mask[0][:-1]
  ## add construction costs for all of these new build years
  proj_cf[new_build_mask] = life_cf[0] * tax_mult * np.power(infl_rate, -1*years[new_build_mask])
  #print(proj_cf)
  ## this is all the years in which decomissioning happens
  ### note that the [0] index is sort of a dummy dimension to help the numpy handshakes
  ### if last decomission is within project life, include that too
  if operating_years[-1] < years[-1]:
    decomission_mask[0] = np.hstack((decomission_mask[0],np.atleast_1d(operating_years[-1]+1)))
  proj_cf[decomission_mask] += life_cf[-1] * tax_mult * np.power(infl_rate, -1*years[decomission_mask])
  #print(proj_cf)
  ## handle the non-build operational years
  non_build_mask = [a[relative_operation!=0] for a in np.where(operating_mask)]
  proj_cf[non_build_mask] += life_cf[relative_operation[relative_operation!=0]] * tax_mult * np.power(infl_rate, -1*years[non_build_mask])
  return proj_cf

def npv_search(settings, components, cash_flows, project_length, v=100):
  """
    Performs NPV matching search
    TODO is the target value required to be 0?
    @ In, settings, CashFlows.GlobalSettings, global settings
    @ In, components, list, list of CashFlows.Component instances
    @ In, cash_flows, dict, component: cashflow: np.array of annual economic values
    @ In, project_length, int, project years
    @ In, v, int, verbosity level
    @ Out, mult, float, multiplier that causes the NPV to match the target value
  """
  m = 'npv search'
  multiplied = 0.0 # cash flows that are meant to include the multiplier
  others = 0.0 # cash flows without the multiplier
  years = np.arange(project_length)
  for comp in components:
    for cf in comp.get_cashflows():
      data = cash_flows[comp.name][cf.name]
      discount_rates = np.power(1.0 + settings.get_discount_rate(), years)
      discounted = np.sum(data/discount_rates)
      if cf.is_mult_target():
        multiplied += discounted
      else:
        others += discounted
  target_val = settings.get_metric_target()
  mult = (target_val - others)/multiplied # TODO div zero possible?
  vprint(v, 0, m, '... NPV multiplier: {: 1.9e}'.format(mult))
  # SANITY CHECL -> FCFF with the multiplier, re-calculate NPV
  if v < 1:
    npv = NPV(components, cash_flows, project_length, settings.get_discount_rate(), mult=mult, v=v)
    if npv != target_val:
      vprint(v, 1, m, 'NPV mismatch warning! Calculated NPV with mult: {: 1.9e}, target: {: 1.9e}'.format(npv, target_val))
  return mult

def FCFF(components, cash_flows, project_length, mult=None, v=100):
  """
    Calculates "free cash flow to the firm" (FCFF)
    @ In, settings, CashFlows.GlobalSettings, global settings
    @ In, cash_flows, dict, component: cashflow: np.array of annual economic values
    @ In, project_length, int, project years
    @ In, mult, float, optional, if provided then scale target cash flow by value
    @ In, v, int, verbosity level
    @ Out, fcff, float, free cash flow to the firm
  """
  m = 'FCFF'
  # FCFF_R for each year
  fcff = np.zeros(project_length)
  for comp in components:
    for cf in comp.get_cashflows():
      data = cash_flows[comp.name][cf.name]
      if mult is not None and cf.is_mult_target():
        fcff += data * mult
      else:
        fcff += data
  vprint(v, 1, m, 'FCFF yearly (not discounted):\n{}'.format(fcff))
  return fcff

def NPV(components, cash_flows, project_length, discount_rate, mult=None, v=100, return_fcff=False):
  """
    Calculates net present value of cash flows
    @ In, components, list, list of CashFlows.Component instances
    @ In, cash_flows, dict, component: cashflow: np.array of annual economic values
    @ In, project_length, int, project years
    @ In, discount_rate, float, firm discount rate to use in discounting future dollars value
    @ In, mult, float, optional, if provided then scale target cash flow by value
    @ In, return_fcff, bool, optional, if True then provide calculated FCFF as well
    @ In, v, int, verbosity level
    @ Out, npv, float, net-present value of system
    @ Out, fcff, float, optional, free cash flow to the firm for same system
  """
  m = 'NPV'
  fcff = FCFF(components, cash_flows, project_length, mult=mult, v=v)
  npv = np.npv(discount_rate, fcff)
  vprint(v, 0, m, '... NPV: {: 1.9e}'.format(npv))
  if not return_fcff:
    return npv
  else:
    return npv, fcff

def IRR(components, cash_flows, project_length, v=100):
  """
    Calculates internal rate of return for system of cash flows
    @ In, components, list, list of CashFlows.Component instances
    @ In, cash_flows, dict, component: cashflow: np.array of annual economic values
    @ In, project_length, int, project years
    @ In, v, int, verbosity level
    @ Out, irr, float, internal rate of return
  """
  m = 'IRR'
  fcff = FCFF(components, cash_flows, project_length, mult=None, v=v) # TODO mult is none always?
  # this method can crash if no solution exists!
  #try:
  irr = np.irr(fcff)
  vprint(v, 1, m, '... IRR: {: 1.9e}'.format(irr))
  #except: # TODO what kind of crash? General catching is bad practice.
  #  vprint(v, 99, m, 'IRR search failed! No solution found. Setting IRR to -10 for debugging.')
  #  irr = -10.0
  return irr

def PI(components, cash_flows, project_length, discount_rate, mult=None, v=100):
  """
    Calculates the profitability index for system
    @ In, components, list, list of CashFlows.Component instances
    @ In, cash_flows, dict, component: cashflow: np.array of annual economic values
    @ In, project_length, int, project years
    @ In, discount_rate, float, firm discount rate to use in discounting future dollars value
    @ In, mult, float, optional, if provided then scale target cash flow by value
    @ In, v, int, verbosity level
    @ Out, pi, float, profitability index
  """
  m = 'PI'
  npv, fcff = NPV(components, cash_flows, project_length, discount_rate, mult=mult, v=v, return_fcff=True)
  pi = -1.0 * npv / fcff[0] # yes, really! This seems strange, but it also seems to be right.
  vprint(v, 1, m, '... PI: {: 1.9e}'.format(pi))
  return pi

def gcd(a, b):
  """
    Find greatest common denominator
    @ In, a, int, first value
    @ In, b, int, sescond value
    @ Out, gcd, int, greatest common denominator
  """
  while b:
    a, b = b, a % b
  return a

def lcm(a, b):
  """
    Find least common multiple
    @ In, a, int, first value
    @ In, b, int, sescond value
    @ Out, lcm, int, least common multiple
  """
  return a * b // gcd(a, b)

def lcmm(*args):
  """
    Find the least common multiple of many values
    @ In, args, list, list of integers to find lcm for
    @ Out, lcmm, int, least common multiple of collection
  """
  return functools.reduce(lcm, args)

#=====================
# MAIN METHOD
#=====================
def run(settings, components, variables):
  """
    @ In, settings, CashFlows.GlobalSettings, global settings
    @ In, components, list, list of CashFlows.Component instances
    @ In, variables, dict, variables from RAVEN
    @ Out, results, dict, economic metric results
  """
  # make a dictionary mapping component names to components
  comps_by_name = dict((c.name, c) for c in components)
  v = settings._verbosity
  m = 'run'
  vprint(v, 0, m, 'Starting CashFlow Run ...')
  # check mapping of drivers and determine order in which they should be evaluated
  vprint(v, 0, m, '... Checking if all drivers present ...')
  ordered = check_drivers(settings, components, variables, v=v)

  # compute project cashflows
  ## this comes in multiple styles!
  ## -> for the "capex/amortization" cashflows, as follows:
  ##    - compute the COMPONENT LIFE cashflow for the component
  ##    - loop the COMPONENT LIFE cashflow until it's as long as the PROJECT LIFE cashflow
  ## -> for the "recurring" sales-type cashflow, as follows:
  ##    - there should already be enough information for the entire PROJECT LIFE
  ##    - if not, and there's only one entry, repeat that entry for the entire project life
  vprint(v, 0, m, '='*90)
  vprint(v, 0, m, 'Component Lifetime Cashflow Calculations')
  vprint(v, 0, m, '='*90)
  lifetime_cashflows = defaultdict(dict) # keys are component, cashflow, then indexed by lifetime
  for ocf in ordered:
    if ocf in variables or ocf == 'EndNode': # TODO why this check for ocf in variables? Should it be comp, or cf?
      continue
    comp_name, cf_name = ocf.split('|')
    comp = comps_by_name[comp_name]
    print('DEBUGG getting', cf_name)
    cf = comp.get_cashflow(cf_name)
    # if this component is a "recurring" type, then we don't need to do the lifetime cashflow bit
    #if cf.type == 'Recurring':
    #  raise NotImplementedError # FIXME how to do this right?
    # calculate cash flow for component's lifetime for this cash flow
    print('jz is comp_name',comp_name)
    print('jz is cf_name',cf_name)
    life_cf = component_life_cashflow(comp, cf, variables, lifetime_cashflows, v=0)

    print('jz is lifecf',life_cf)
    lifetime_cashflows[comp_name][cf_name] = life_cf

  vprint(v, 0, m, '='*90)
  vprint(v, 0, m, 'Project Lifetime Cashflow Calculations')
  vprint(v, 0, m, '='*90)
  # determine how the project life is calculated.
  project_length = get_project_length(settings, components, v=v)
  vprint(v, 0, m, ' ... project length: {} years'.format(project_length))
  project_cashflows = project_life_cashflows(settings, components, lifetime_cashflows, project_length, v=v)

  vprint(v, 0, m, '='*90)
  vprint(v, 0, m, 'Economic Indicator Calculations')
  vprint(v, 0, m, '='*90)
  indicators = settings.get_indicators()
  results = {}
  if 'NPV_search' in indicators:
    metric = npv_search(settings, components, project_cashflows, project_length, v=v)
    results['NPV_mult'] = metric
  if 'NPV' in indicators:
    metric = NPV(components, project_cashflows, project_length, settings.get_discount_rate(), v=v)
    results['NPV'] = metric
  if 'IRR' in indicators:
    metric = IRR(components, project_cashflows, project_length, v=v)
    results['IRR'] = metric
  if 'PI' in indicators:
    metric = PI(components, project_cashflows, project_length, settings.get_discount_rate(), v=v)
    results['PI'] = metric
  return results


#=====================
# PRINTING STUFF
#=====================
def vprint(threshold, desired, method, *msg):
  """
    Light wrapper for printing that considers verbosity levels
    @ In, threshold, int, cutoff verbosity
    @ In, desired, int, requested message verbosity level
    @ In, method, str, name of method raising print
    @ In, msg, list(str), messages to print
    @ Out, None
  """
  if desired >= threshold:
    print('CashFlow INFO ({}):'.format(method), *msg)
