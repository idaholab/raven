# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on Feb 4, 2016

Module aimed to define the methods to group variables in the RAVEN frameworl
"""

#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
from collections import OrderedDict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import BaseClasses

#Internal Modules End--------------------------------------------------------------------------------


class VariableGroup(BaseClasses.BaseType):
  """
    Allows grouping of variables for ease of access
  """

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    BaseClasses.BaseType.__init__(self)
    self.printTag = 'VariableGroup'
    self._dependents = []  #name of groups this group's construction is dependent on
    self._base = None  #if dependent, the name of base group to start from
    self._list = []  #text from node
    self.variables = []  #list of variable names
    self.initialized = False  #true when initialized

  def _readMoreXML(self, node):
    """
      reads XML for more information
      @ In, node, xml.etree.ElementTree.Element, xml element to read data from
      @ Out, None
    """
    #establish the name
    if 'name' not in node.attrib.keys():
      self.raiseAnError(IOError, 'VariableGroups require a "name" attribute!')
    self.name = node.attrib['name']
    #dependents
    deps = node.attrib.get('dependencies', None)
    if deps is not None and len(deps) > 0:
      if 'base' not in node.attrib.keys():
        self.raiseAnError(IOError,
                          'VariableGroups with dependencies require a "base" group to start from!')
      self._base = node.attrib.get('base')
      self._dependents = list(g.strip() for g in deps.split(','))
    self._list = node.text.split(',')

  def initialize(self, varGroups):
    """
      Establish variable set.
      @ In, varGroups, list, VariableGroup classes
      @ Out, None
    """
    if len(self._dependents) == 0:
      self.variables = list(
          l.strip() for l in self._list
      )  #set(l.strip() for l in self._list) #don't use sets, since they destroy order
    else:
      #get base
      base = None
      for group in varGroups:
        if group.name == self._base:
          base = group
          break
      if base is None:
        self.raiseAnError(IOError, 'Base %s not found among variable groups!' % self._base)
      #get dependencies
      deps = OrderedDict()
      for depName in self._dependents:
        dep = None
        for group in varGroups:
          if group.name == depName:
            dep = group
            break
        if dep is None:
          self.raiseAnError(IOError, 'Dependent %s not found among variable groups!' % depName)
        deps[depName] = dep
      #get base set
      baseVars = set(base.getVars())
      #do modifiers to base
      modifiers = list(m.strip() for m in self._list)
      orderOps = []  #order of operations that occurred, just var names and dep lists
      for mod in modifiers:
        #remove internal whitespace
        mod = mod.replace(' ', '')
        #get operator and varname
        op = mod[0]
        varName = mod[1:]
        if op not in ['+', '-', '^', '%']:
          self.raiseAnError(IOError, 'Unrecognized or missing dependency operator:', op, varName)
        #if varName is a single variable, make it a set so it behaves like the rest
        if varName not in deps.keys():
          modSet = [varName]
        else:
          modSet = deps[varName].getVars()
        orderOps.append(modSet[:])
        modSet = set(modSet)
        if op == '+':
          baseVars.update(modSet)
        elif op == '-':
          baseVars.difference_update(modSet)
        elif op == '^':
          baseVars.intersection_update(modSet)
        elif op == '%':
          baseVars.symmetric_difference_update(modSet)
      #sort variable list into self.variables
      #  -> first, sort through top-level vars
      for var in base.getVars():
        if var in baseVars:
          self.variables.append(var)
          baseVars.remove(var)
      #  -> then, sort through deps/operations in order
      for mod in orderOps:
        for var in mod:
          if var in baseVars:
            self.variables.append(var)
            baseVars.remove(var)
      #  -> baseVars better be empty now!
      if len(baseVars) > 0:
        self.raiseAWarning('    End vars    :', self.variables)
        self.raiseAWarning('    End BaseVars:', baseVars)
        self.raiseAnError(
            RuntimeError,
            'Not all variableGroup entries were accounted for!  The operations were not performed correctly'
        )
    self.initialized = True

  def getDependencies(self):
    """
      Returns list object of strings containing variable group names
      @ In, None
      @ Out, _dependents, list(str), list of variable group names
    """
    return self._dependents[:]

  def getVars(self):
    """
      Returns set object of strings containing variable names in group
      @ In, None
      @ Out, variables, list(str), set of variable names
    """
    return self.variables[:]

  def getVarsString(self, delim=','):
    """
      Returns delim-separated list of variables in group
      @ In, delim, string, optional, delimiter (default = ',')
      @ Out, csvVariablesString, string, list of variables in comma-separated string
    """
    csvVariablesString = ','.join(self.getVars())
    return csvVariablesString


#
#
#
# end
