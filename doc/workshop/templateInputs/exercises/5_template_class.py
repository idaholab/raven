import os
import sys

ravenTemplateDir = os.path.join(os.path.dirname(__file__), '../'*4, 'framework')
sys.path.append(ravenTemplateDir)

from InputTemplates.TemplateBaseClass import Template as TemplateBase
from utils import xmlUtils

class ProjectileTemplateClass(TemplateBase):
  """ Input writer for Workshop Projectile UQ Template demonstration """

  # optional: add naming templates that can be used later.
  TemplateBase.addNamingTemplates({'distribution': '{var}_dist'})
  varDefaultValues = {'v0': 30, 'y0': 0, 'angle': 45}

  def createWorkflow(self, inputs):
    """
      creates a new RAVEN workflow based on the information in dicitonary "inputs".
      "inputs" keys can contain the variables [v0, y0, angle] mapped to distribution dictionaries, or
        it can contain "metrics" mapped to any combination of [mean, std, percentile]
      @ In, inputs, dict, info that was read from user input
      @ Out, xml.etree.ElementTree.Element, modified copy of template ready to run
    """
    # call the base class to read in the template; this just creates a copy of the XML tree in self._template.
    template = TemplateBase.createWorkflow(self)
    # loop over the variables and modify the XML for each of them
    for var in self.varDefaultValues.keys():
      distData = inputs.get(var, None)
      template = self._modifyVariables(template, var, distData)
    # which variables are actually uncertain?
    uncertain = []
    for var in self.varDefaultValues.keys():
      if inputs[var]['dist'] in ['uniform', 'normal']:
        uncertain.append(var)
    # for uncertain variables, apply metrics
    template = self._applyMetrics(template, uncertain, inputs.get('metric', None))
    return template

  def _modifyVariables(self, template, var, distData):
    """
      Modifies template for the requested variable
      @ In, template, xml.etree.ElementTree.Element, root of template XML
      @ In, var, string, one of the variable names
      @ In, distData, dict, distribution information (or None if none given)
      @ Out, template, xml.etree.ElementTree.Element, modified template
    """
    # if distribution data is None, replace it with the default constant value
    if distData is None:
      distData = {'dist': 'constant', 'value': self.varDefaultValues[var]}

    # modify the Distributions
    distributions = template.find('Distributions')
    if distData['dist'] == 'uniform':
      name = self.namingTemplates['distribution'].format(var=var)
      lower = distData['lowerBound']
      upper = distData['upperBound']
      distNode = xmlUtils.newNode('Uniform', attrib={'name': name})
      distNode.append(xmlUtils.newNode('lowerBound', text=lower))
      distNode.append(xmlUtils.newNode('upperBound', text=upper))
      distributions.append(distNode)
    elif distData['dist'] == 'normal':
      name = self.namingTemplates['distribution'].format(var=var)
      mean = distData['mean']
      sigma = distData['std']
      lower = distData.get('lowerBound', None)
      upper = distData.get('upperBound', None)
      distNode = xmlUtils.newNode('Normal', attrib={'name': name})
      distNode.append(xmlUtils.newNode('mean', text=mean))
      distNode.append(xmlUtils.newNode('sigma', text=sigma))
      if lower is not None:
        distNode.append(xmlUtils.newNode('lowerBound', text=lower))
      if upper is not None:
        distNode.append(xmlUtils.newNode('upperBound', text=upper))
      distributions.append(distNode)
    else:
      pass # no distribution needed for Constant variables

    # modify Sampler
    sampler = template.find('Samplers').find('MonteCarlo')
    if distData['dist'] == 'constant':
      sampler.append(xmlUtils.newNode('constant', attrib={'name': var}, text=distData['value']))
    else:
      varNode = xmlUtils.newNode('variable', attrib={'name': var})
      varNode.append(xmlUtils.newNode('distribution', text=self.namingTemplates['distribution'].format(var=var)))
      sampler.append(varNode)

    return template

  def _applyMetrics(self, template, uncertain, metrics):
    """
      sets up metrics for statistical postprocessing
      @ In, template, xml.etree.ElementTree.Element, root of template XML
      @ In, uncertain, list, list of variables that are uncertain
      @ In, metrics, list, list of metrics to use (or None if none given)
      @ Out, template, xml.etree.ElementTree.Element, modified root of template XML
    """
    if metrics is None:
      metrics = ['mean']
    uncVars = ','.join(uncertain)

    ## add to PostProcessor, also output DataObject
    # find the postprocessor
    statsPP = template.find('Models').find('PostProcessor')
    # find the stats data object
    dataObjs = template.find('DataObjects').findall('PointSet')
    for dataObj in dataObjs:
      if dataObj.attrib['name'] == 'stats':
        break
    outputsNode = dataObj.find('Output')
    for metric in metrics:
      if metric == 'mean':
        statsPP.append(xmlUtils.newNode('expectedValue', attrib={'prefix':'mean'}, text=uncVars))
        for var in uncertain:
          self._updateCommaSeperatedList(outputsNode, '{}_{}'.format(metric, var))
      elif metric == 'std':
        statsPP.append(xmlUtils.newNode('sigma', attrib={'prefix':'std'}, text=uncVars))
        for var in uncertain:
          self._updateCommaSeperatedList(outputsNode, '{}_{}'.format(metric, var))
      elif metric == 'percentile':
        statsPP.append(xmlUtils.newNode('percentile', attrib={'prefix':'pct'}, text=uncVars))
        for var in uncertain:
          self._updateCommaSeperatedList(outputsNode, '{}_{}'.format('pct_5', var))
          self._updateCommaSeperatedList(outputsNode, '{}_{}'.format('pct_95', var))
    return template
