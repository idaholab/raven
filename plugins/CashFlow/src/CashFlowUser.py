
import os
import sys

# NOTE this import exception is ONLY to allow RAVEN to directly import this module.
try:
  from CashFlow.src.CashFlows import Component
except ImportError:
  from CashFlows import Component

# This plugin imports RAVEN modules. if run in stand-alone, RAVEN needs to be installed and this file
# needs to be in the propoer plugin directory.
raven_path = os.path.dirname(__file__) + '/../../../framework'
sys.path.append(os.path.expanduser(raven_path))
from utils import InputData

class CashFlowUser:
  """
    Base class for objects that want to access the functionality of the CashFlow objects.
    Generally this means the CashFlowUser will have an "economics" xml node used to define it,
    and will have a group of cash flows associated with it (e.g. a "component")

    In almost all cases, initialization methods should be called as part of the inheritor's method call.
  """
  @classmethod
  def get_input_specs(cls, spec):
    """
      Collects input specifications for this class.
      Note this needs to be called as part of an inheriting class's specification definition
      @ In, spec, InputData, specifications that need cash flow added to it
      @ Out, input_specs, InputData, specs
    """
    # this unit probably has some economics
    spec.addSub(Component.get_input_specs())
    return spec

  def __init__(self):
    """
      Constructor
      @ In, kwargs, dict, optional, arguments to pass to other constructors
      @ Out, None
    """
    self._economics = None # CashFlowGroup

  def read_input(self, specs):
    """
      Sets settings from input file
      @ In, specs, InputData params, input from user
      @ Out, None
    """
    self._economics = Component(self)
    self._economics.read_input(specs)

  def get_crossrefs(self):
    """
      Collect the required value entities needed for this component to function.
      @ In, None
      @ Out, crossrefs, dict, mapping of dictionaries with information about the entities required.
    """
    return self._economics.get_crossrefs()

  def set_crossrefs(self, refs):
    """
      Connect cross-reference material from other entities to the ValuedParams in this component.
      @ In, refs, dict, dictionary of entity information
      @ Out, None
    """
    self._economics.set_crossrefs(refs)

  def get_incremental_cost(self, activity, raven_vars, meta, t):
    """
      get the cost given particular activities
      @ In, activity, pandas.Series, scenario variable values to evaluate cost of
      @ In, raven_vars, dict, additional variables (presumably from raven) that might be needed
      @ In, meta, dict, further dictionary of information that might be needed
      @ In, t, int, time step at which cost needs to be evaluated
      @ Out, cost, float, cost of activity
    """
    return self._economics.incremental_cost(activity, raven_vars, meta, t)

  def get_economics(self):
    """
      Accessor for economics.
      @ In, None
      @ Out, econ, CashFlowGroup, cash flows for this cash flow user
    """
    return self._economics
