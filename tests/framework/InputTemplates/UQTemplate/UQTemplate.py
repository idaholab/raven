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

This module is an exemplary template for doing UQ calculations.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)
# standard library
import os
import sys
# external libraries
# RAVEN libraries
## caution: using abspath here does not do desirable things on Windows Mingw.
frameworkDir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'framework'))
sys.path.append(frameworkDir)
from utils import xmlUtils
from InputTemplates.TemplateBaseClass import Template



class UQTemplate(Template):
  """
    Generic class for templating input files.
    Intended to be used to read a template, be given instructions on how to fill it,
    and create a new set of input files.
  """
  # generic class members
  Template.addNamingTemplates({'step name'   : '{action}_{subject}',
                               'distribution': '{var}_dist',
                               'metric var'  : '{metric}_{var}',
                              })
  metrics = ['expectedValue', 'sigma', 'skewness', 'kurtosis']

  ###############
  # API METHODS #
  ###############
  def createWorkflow(self, model=None, variables=None, samples=None, case=None, **kwargs):
    """
      Creates a new RAVEN workflow file based on the information in kwargs.
      Specific to individual templates. Must overload to be useful.
      @ In, model, dict, information about the model
      @ In, variables, dict, information about the variables
      @ In, samples, int, number of samples to take
      @ In, kwargs, dict, other unused keyword arguments
      @ Out, xml.etree.ElementTree.Element, modified copy of template
    """
    template = Template.createWorkflow(self, **kwargs)
    # working dir
    template.find('RunInfo').find('WorkingDir').text = case
    # populate varible list
    allVars = list(variables.keys()) + model['output']
    # adjust external model block
    extModelNode = template.find('Models').find('ExternalModel')
    extModelNode.attrib['ModuleToLoad'] = model['file']
    extModelNode.find('variables').text = ', '.join(allVars)
    # handle variables
    for var, info in variables.items():
      self._addInputVariable(template, var, info['mean'], info['std'])
    # also output variables
    for var in model['output']:
      self._addOutputVariable(template, var)
    # number of samples
    template.find('Samplers').find('MonteCarlo').find('samplerInit').find('limit').text = str(samples)
    return template


  ################################
  # INPUT CONSTRUCTION SHORTCUTS #
  ################################
  def _addInputVariable(self, xml, varName, mean, std):
    """
      Add a SAMPLED variable everywhere needed in the calculation workflow.
      @ In, xml, xml.etree.ElementTree.Element, Simulation node from template
      @ In, varName, str, name of variable for whom distribution is being created
      @ In, mean, float, desired mean of distribution
      @ In, std, float, desired standard deviation of distribution
      @ Out, None
    """
    # add the distribution
    self._normalDistribution(xml, varName, mean, std)
    # add variable to sampler
    self._samplerVariable(xml.find('Samplers').find('MonteCarlo'), varName)
    # add variable to data object
    datas = xml.find('DataObjects')
    ## NOTE relying on template order: 0 is placeholder, 1 is samples, 2 is stats
    self._updateCommaSeperatedList(datas[0].find('Input'), varName)
    self._updateCommaSeperatedList(datas[1].find('Input'), varName)
    for metric in self.metrics:
      self._updateCommaSeperatedList(datas[2].find('Output'), self.namingTemplates['metric var'].format(metric=metric, var=varName))
    # add to basic statistics request
    stats = xml.find('Models').find('PostProcessor')
    for metric in self.metrics:
      self._updateCommaSeperatedList(stats.find(metric), varName)

  def _addOutputVariable(self, xml, varName):
    """
      Add an OUTPUT variable everywhere needed in the calculation workflow.
      @ In, xml, xml.etree.ElementTree.Element, Simulation node from template
      @ In, varName, str, name of variable for whom distribution is being created
      @ Out, None
    """
    # data objects
    datas = xml.find('DataObjects')
    ## NOTE relying on template order: 0 is placeholder, 1 is samples, 2 is stats
    self._updateCommaSeperatedList(datas[1].find('Output'), varName)
    for metric in self.metrics:
      self._updateCommaSeperatedList(datas[2].find('Output'), self.namingTemplates['metric var'].format(metric=metric, var=varName))
    # add to basic statistics request
    stats = xml.find('Models').find('PostProcessor')
    for metric in self.metrics:
      self._updateCommaSeperatedList(stats.find(metric), varName)


  def _samplerVariable(self, xml, varName):
    """
      Adds a variable block to a Sampler
      @ In, xml, xml.etree.ElementTree.Element, specific Sampler node to add to
      @ In, varName, str, name of variable
      @ Out, None
    """
    var = xmlUtils.newNode('variable', attrib={'name':varName})
    var.append(xmlUtils.newNode('distribution', text=self.namingTemplates['distribution'].format(var=varName)))
    xml.append(var)

  def _normalDistribution(self, xml, varName, mean, std):
    """
      Adds a normal distribution to the Distributions block
      @ In, xml, xml.etree.ElementTree.Element, Simulation node from template
      @ In, varName, str, name of variable for whom distribution is being created
      @ In, mean, float, desired mean of distribution
      @ In, std, float, desired standard deviation of distribution
      @ Out, None
    """
    dist = xmlUtils.newNode('Normal', attrib={'name':self.namingTemplates['distribution'].format(var=varName)})
    dist.append(xmlUtils.newNode('mean', text=str(float(mean))))
    dist.append(xmlUtils.newNode('sigma', text=str(float(std))))
    xml.find('Distributions').append(dist)
