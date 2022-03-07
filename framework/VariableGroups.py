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
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
from collections import OrderedDict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from . import BaseClasses
#Internal Modules End--------------------------------------------------------------------------------

#
#
#
#
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
    self.printTag       = 'VariableGroup'
    self.variables      = []             #list of variable names
    self.initialized    = False          #true when initialized

  def readXML(self, node, varGroups):
    """
      reads XML for more information
      @ In, node, xml.etree.ElementTree.Element, xml element to read data from
      @ In, varGroups, dict, other variable groups including ones this depends on (if any)
      @ Out, None
    """
    #establish the name
    if 'name' not in node.attrib.keys():
      self.raiseAnError(IOError,'VariableGroups require a "name" attribute!')
    self.name = node.attrib['name']
    if node.text is None:
      node.text = ''
    # loop through variables and expand list
    for dep in [s.strip() for s in node.text.split(',')]:
      if dep == '':
        continue
      # get operator if provided
      operator = '+'
      if dep[0] in '+-^%':
        operator = dep[0]
        dep = dep[1:].strip()
      # expand variables if a group name is given
      if dep in varGroups:
        dep = varGroups[dep].getVars()
      else:
        dep = [dep]

      # apply operators
      toRemove = []
      ## union
      if operator == '+':
        for d in dep:
          if d not in self.variables:
            self.variables.append(d)
      ## difference
      elif operator == '-':
        for d in dep:
          try:
            self.variables.remove(d)
          except ValueError:
            self.raiseADebug('Was asked to remove "{}" from variable group "{}", but it is not present! Ignoring ...'
                               .format(d,self.name))
      ## intersection
      elif operator == '^':
        for v in self.variables:
          if v not in dep:
            toRemove.append(v)
      ## symmetric difference
      elif operator == '%':
        for v in self.variables:
          if v in dep:
            toRemove.append(v)
        for d in dep:
          if d not in self.variables:
            self.variables.append(d)
      ## cleanup
      for v in toRemove:
        self.variables.remove(v)

    # finished
    self.raiseADebug('Variable group "{}" includes:'.format(self.name),self.getVarsString())


  def getVars(self):
    """
      Returns set object of strings containing variable names in group
      @ In, None
      @ Out, variables, list(str), set of variable names
    """
    return self.variables[:]

  def getVarsString(self,delim=','):
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
#
