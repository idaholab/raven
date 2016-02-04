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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
import utils
import BaseClass
#Internal Modules End--------------------------------------------------------------------------------


class VariableGroup(BaseClass):
  """
    Allows grouping of variables for ease of access
  """
  def __init__(self,messageHandler):
    """
      Constructor
    """
    self.messageHandler = messageHandler #message handling tool
    self.name           = ''             #identifier
    self.variables      = set()          #list of variable names

  def readMoreXML(self,node):
    """
      reads XML for more information
      @ In, node, ET.Element, xml element to read data from
      @ Out, None
    """
    #establish the name
    if 'name' not in node.attrib.keys():
      self.raiseAnError(IOError,'VariableGroups require a "name" attribute!')
    self.name = node.attrib['name']
    #if a subset, figure that out

  def intialize(self):
    """
      Establish variable set.
      @ In, varGroups, list, VariableGroup classes
      @ Out, None
    """
    self.variables = self.__listVars

  def getVars(self,delim=','):
    """
      Returns delim-separated list of variables in group
      @ In, delim, string, delimiter
    """
    return ','.join(self.variables)
#
#
#
#
class Subset(VariableGroup):
  """
    Subset of a VariableGroup, only different in construction
  """
  def __init__(self,messageHandler):
    """
      Constructor
    """
    VariableGroup.__init__(self,messageHandler)
    self.__supersetName = None           #name of set of whom this group is a subset, optional
    self.__superset     = None           #set of whom this group is a subset, optional
    self.__choice       = None           #method for constructing this subset, optional
    self.__listVars     = None           #variables listed in input xml node

  def readMoreXML(self,node):
    """
      reads XML for more information
      @ In, node, ET.Element, xml element to read data from
      @ Out, None
    """
    VariableGroup.readMoreXML(self,node)
    if 'superset' not in node.attrib.keys():
      self.raiseAnError(IOError,'No superset specified for subset variable group!')
    if 'choice' not in node.attrib.keys():
      self.raiseAnError(IOError,'No choice method specified for subset variable group!')
    self.__supersetName = node.attrib['superset']
    self.__choice       = node.attrib['choice']
    if self.__choice not in ['include','exclude']:
      self.raiseAnError(IOError,'Specified choice method not recognized:',self.__choice)
    self.__listVars     = list(v.strip() for v in node.text.split(','))

  def intialize(self,varGroups):
    """
      Establish variable set.
      @ In, varGroups, list, VariableGroup classes
      @ Out, None
    """
    #find superset
    for group in varGroups:
      if group.name == self.__supersetName:
        self.__superset = group
        break
    if self.__superset is None:
      self.raiseAnError(IOError,'No matching variable group found among Groups:',self.__supersetName)
    #construct subset based on choice
    if self.__choice == 'include': #note: this seems redundant with just making a new VariableGroup right now.  Does it have a use?
      self.variables = self.__listVars
    elif self.__choice == 'exclude':
      self.variables = self.__superset.variables[:]
      for v in self.__listVars:
        while v in self.variables:
          self.variables.remove(v)





