#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import os
import time
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
import BaseClasses
#Internal Modules End--------------------------------------------------------------------------------


class VariableGroup(BaseClasses.BaseType):
  """
    Allows grouping of variables for ease of access
  """
  def __init__(self):
    """
      Constructor
    """
    BaseClasses.BaseType.__init__(self)
    self.printTag       = 'VariableGroup'
    self._dependents    = []             #name of groups this group's construction is dependent on
    self._base          = None           #if dependent, the name of base group to start from
    self._list          = []             #text from node
    self.variables      = set()          #list of variable names
    self.initialized    = False          #true when initialized

  def _readMoreXML(self,node):
    """
      reads XML for more information
      @ In, node, ET.Element, xml element to read data from
      @ Out, None
    """
    #establish the name
    if 'name' not in node.attrib.keys():
      self.raiseAnError(IOError,'VariableGroups require a "name" attribute!')
    self.name = node.attrib['name']
    #dependents
    deps = node.attrib.get('dependencies',None)
    if deps is not None and len(deps)>0:
      if 'base' not in node.attrib.keys():
        self.raiseAnError(IOError,'VariableGroups with dependencies require a "base" group to start from!')
      self._base = node.attrib.get('base')
      self._dependents = list(g.strip() for g in deps.split(','))
    self._list = node.text.split(',')

  def initialize(self,varGroups):
    """
      Establish variable set.
      @ In, varGroups, list, VariableGroup classes
      @ Out, None
    """
    if len(self._dependents)==0:
      self.variables = set(l.strip() for l in self._list)
    else:
      #get base
      base = None
      for group in varGroups:
        if group.name==self._base:
          base = group
          break
      if base is None:
        self.raiseAnError(IOError,'Base %s not found among variable groups!' %self._base)
      #get dependencies
      deps={}
      for depName in self._dependents:
        dep = None
        for group in varGroups:
          if group.name==depName:
            dep = group
            break
        if dep is None:
          self.raiseAnError(IOError,'Dependent %s not found among variable groups!' %depName)
        deps[depName] = dep
      #get base set
      baseVars = base.getVars()
      #do modifiers to base
      modifiers = list(m.strip() for m in self._list)
      for mod in modifiers:
        #remove internal whitespace
        mod = mod.replace(' ','')
        #get operator and varname
        op = mod[0]
        varName = mod[1:]
        if op not in ['+','-','^','%']:
          self.raiseAnError(IOError,'Unrecognized or missing dependency operator:',op,varName)
        #if varName is a single variable, make it a set so it behaves like the rest
        if varName not in deps.keys():
          modset = set([varName])
        else:
          modset = deps[varName].getVars()
        if   op == '+': baseVars.update(modset)
        elif op == '-': baseVars.difference_update(modset)
        elif op == '^': baseVars.intersection_update(modset)
        elif op == '%': baseVars.symmetric_difference_update(modset)
        #set class variable
      self.variables = baseVars
    self.initialized=True

  def getDependencies(self):
    """
      Returns list object of strings containing variable group names
      @ In, None
      @ Out, list(str), list of variable group names
    """
    return self._dependents[:]

  def getVars(self):
    """
      Returns set object of strings containing variable names in group
      @ In, None
      @ Out, set(str), set of variable names
    """
    return self.variables.copy()

  def getVarsString(self,delim=','):
    """
      Returns delim-separated list of variables in group
      @ In, delim, string, delimiter
    """
    return ','.join(self.getVars())
#
#
#
# end
