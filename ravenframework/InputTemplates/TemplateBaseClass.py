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
Created on February 22 2019
@author: talbpaul

This module is the base class for Input Templates, which use an established input
template as an accelerated way to write new RAVEN workflows. Other templates
can inherit from this base class for specific applications.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
# standard library
import os
import sys
import copy
# external libraries
# RAVEN libraries
from ..utils import xmlUtils
from ..BaseClasses import MessageUser


class Template(MessageUser):
  """
    Generic class for templating input files.
    Intended to be used to read a template, be given instructions on how to fill it,
    and create a new set of input files.
  """
  # naming templates
  ## this dictionary of templates can be used to standardize naming throughout a template.
  ## for example, variables in a template might always have the form source_contents, e.g.
  ## "model_samples" or "rom_meta". In this case, a standardized naming format might look
  ## like '{source}_{content}', so that the variables can be consistently named as
  ## self.namingTemplates['variable'].format(source='model', content='samples'). Register
  ## new templates by calling (for example) self.addNamingTemplate({'variable':'{source}_{content}'})
  ## This is an optional tool, and in no way needs to be used to build templates.
  namingTemplates = {} # Updated in inheriting classes by calling cls.addNamingTemplates


  @classmethod
  def addNamingTemplates(cls, templates):
    """
      Extends naming conventions with the provided templates.
      @ In, templates, dict, new formatted templates to use
      @ Out, None
    """
    cls.namingTemplates.update(templates)


  ###############
  # API METHODS #
  ###############
  def __init__(self, **kwargs):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._template = None     # XML element with the root Simulation node of a RAVEN input
    # assure that the template path gives the location of the inheriting template, not the base class
    self._templatePath = os.path.dirname(os.path.normpath(sys.modules[self.__class__.__module__].__file__))

  def createWorkflow(self, **kwargs):
    """
      Creates a new RAVEN workflow file based on the information in kwargs.
      Specific to individual templates. Must overload to be useful.
      @ In, kwargs, dict, keyword arguments
      @ Out, xml.etree.ElementTree.Element, modified copy of template
    """
    # by default, don't do anything special.
    return copy.deepcopy(self._template)

  def loadTemplate(self, filename, path):
    """
      Loads template file statefully.
      @ In, filename, str, name of file to load (xml)
      @ In, path, str, path (maybe relative) to file
      @ Out, None
    """
    # TODO what should "path" be relative to? I vote the Template file.
    relPath = os.path.join(self._templatePath, path)
    templateFile = os.path.join(os.path.normpath(relPath), filename)
    self._template, _ = xmlUtils.loadToTree(templateFile)

  def writeWorkflow(self, template, destination, run=False):
    """
      Writes a template to file.
      @ In, template, xml.etree.ElementTree.Element, file to write
      @ In, destination, str, path and filename to write to
      @ In, run, bool, optional, if True then run the workflow after writing? good idea?
      @ Out, errors, int, 0 if successfully wrote [and run] and nonzero if there was a problem
    """
    pretty = xmlUtils.prettify(template)
    with open(destination, 'w') as f:
      f.write(pretty)
    if run:
      errors = self.runWorkflow(destination)
    else:
      # 0 meaning successfully wrote file
      errors = 0
    return errors

  def runWorkflow(self, destination):
    """
      Runs the workflow at the destination.
      @ In, destination, str, path and filename of RAVEN input file
      @ Out, res, int, system results of running the code
    """
    # where are we putting the file?
    destDir = os.path.dirname(os.path.normpath(destination))
    workflow = os.path.basename(destination)
    cwd = os.getcwd()
    os.chdir(destDir)
    raven = self._getRavenLocation(which='exec')
    command = '{command} {workflow}'.format(command=raven,
                                            workflow=workflow)
    res = os.system(command)
    os.chdir(cwd)
    return res



  ################################
  # INPUT CONSTRUCTION SHORTCUTS #
  ################################
  def _assemblerNode(self, tag, cls, typ, name):
    """
      Constructs an Assembler-type node given the parameters.
      e.g. <SolutionExport class='DataObjects' type='DataSet'>export</SolutionExport>
      @ In, tag, str, name of node
      @ In, cls, str, name of RAVEN class
      @ In, typ, str, name of RAVEN class's type
      @ In, name, str, RAVEN name of object
      @ Out, node, xml.etree.ElementTree.Element, element formatted RAVEN style.
    """
    attrib = {'class':cls, 'type':typ}
    node = xmlUtils.newNode(tag, attrib=attrib, text=name)
    return node

  def _updateCommaSeperatedList(self, node, new, position=None, before=None, after=None):
    """
      Statefully adds an entry to the given node's comma-seperated text
      If the node's text is empty, will insert "new" as the sole text (no commas).
      @ In, node, xml.etree.ElementTree.Element, node whose text is a comma-seperated string list
      @ In, new, str, name of entry to add
      @ In, position, int, optional, index where new should be inserted in sequence
      @ In, before, str, optional, entry name before which new should be added
      @ In, after, str, optional, entry name after which new should be added
      @ Out, None
    """
    entries = list(x.strip() for x in node.text.split(',')) if node.text is not None else []
    # remove empty strings
    entries = list(filter(None, entries))
    if position is not None:
      if position < len(entries):
        entries.insert(position, new)
      else:
        entries.append(new)
    elif before is not None:
      if before in entries:
        entries.insert(entries.index(before), new)
      else:
        entries.append(new)
    elif after is not None:
      if after in entries:
        entries.insert(entries.index(after)+1, new)
      else:
        entries.append(new)
    else:
      entries.append(new)
    node.text = ', '.join(entries)

  ################################
  # OTHER UTILITIES              #
  ################################
  def _getRavenLocation(self, which='framework'):
    r"""
      Returns the (string) path to RAVEN
      NOTE this is problematic in mingw windows, since using abspath includes e.g. C:\msys64\home\etc.
      @ In, framework, bool, optional, if True then give location of "raven/framework" else "raven/"
      @ Out, path, str, path to raven
    """
    assert which in ['framework', 'exec', 'head']
    if which == 'framework':
      path = frameworkDir
    elif which == 'exec':
      path = os.path.normpath(os.path.join(frameworkDir, '..', 'raven_framework'))
    else:
      path = os.path.normpath(os.path.join(frameworkDir, '..'))
    return path
