from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
import math
import xml.etree.ElementTree as ET
from utils import utils
from utils import graphStructure as GS
import copy 
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginsBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
#Internal Modules End-----------------------------------------------------------


class graphModel(ExternalModelPluginBase):

  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, xmlNode, xml.etree.ElementTree.Element, XML node that needs to be read
      @ Out, None
    """
    container.modelFile = None
    container.nodesIN   = None
    container.nodesOUT  = None

    container.mapping    = {}
    container.InvMapping = {}

    container.calcMode = None

    for child in xmlNode:
      if   child.tag == 'nodesIN':
        container.nodesIN  = [str(var.strip()) for var in child.text.split(",")]
      elif child.tag == 'nodesOUT':
        container.nodesOUT = [str(var.strip()) for var in child.text.split(",")]
      elif child.tag == 'modelFile':
        container.modelFile = child.text.strip() + '.xml'
      elif child.tag == 'map':
        container.mapping[child.get('var')]      = child.text.strip()
        container.InvMapping[child.text.strip()] = child.get('var')
      elif child.tag == 'variables':
        variables = [str(var.strip()) for var in child.text.split(",")]
      elif child.tag == 'timeDep':
        container.timeDepVars = [str(var.strip()) for var in child.text.split(",")]
        if container.timeDepVars:
          container.calcMode = True
        else:
          container.calcMode = False
        if not container.timeDepVars <= set(container.mapping.values()):
          print('timeDepVars error')
      else:
        print('xml error')

    if container.nodesIN is None:
      print('nodesIN Error')
    if container.nodesOUT is None:
      print('nodesOUT Error')
    if container.modelFile is None:
      print('modelFile Error')  

    if set(variables) != set(container.mapping.keys()):
      print('variables error')
    if not set(container.nodesOUT) <= set(container.mapping.values()):
      print('nodesOUT error')

  def initialize(self, container,runInfoDict,inputFiles):
    """
      Method to initialize this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """
    container.nodes = {}  
    container.deg   = {}

    container.runInfo = runInfoDict
    self.createGraph(container,container.modelFile)

  def createGraph(self,container,file):
    graph = ET.parse(container.runInfo['WorkingDir'] + '/' + file)
    graph = findAllRecursive(graph,'Graph')

    for node in findAllRecursive(graph[0], 'node'):
      nodeName = node.get('name')
      nodeChilds = []
      deg = None
      for child in node:
        if child.tag == 'childs':
          nodeChilds = [var.strip() for var in child.text.split(",")]
        if child.tag == 'deg':
          deg = float(child.text)
      container.nodes[nodeName] = nodeChilds
      container.deg[nodeName]   = deg

  def run(self, container, Inputs):
    if container.calcMode:
      self.runTimeDep(container, Inputs)
    else:
      dictOUT = self.runStatic(container, Inputs)
      for var in dictOUT.keys():
        container.__dict__[var] = dictOUT[var]

  def runStatic(self, container, Inputs):
    """
      This is a simple example of the run method in a plugin.
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN

    """
    mapping = copy.deepcopy(container.mapping)
    nodes   = copy.deepcopy(container.nodes)

    for key in Inputs.keys():
      if key in mapping.keys():
        if mapping[key] in nodes.keys() and Inputs[key][0] == 1.0:
          nodes.pop(mapping[key],None)
          for node in nodes.keys():
            if mapping[key] in nodes[node]:
              nodes[node].remove(mapping[key])
    
    ravenGraph = GS.graphObject(nodes)

    dictOUT = {}
    for nodeO in container.nodesOUT:
      paths = []
      for nodeI in container.nodesIN:
        paths = paths + ravenGraph.findAllPaths(nodeI,nodeO)
      var = container.InvMapping[nodeO]
      if paths:
        dictOUT[var] = np.asarray(0.)
      else:
        dictOUT[var] = np.asarray(1.)
    return dictOUT
    
def runTimeDep(self, container, Inputs):
  pass
  '''
  times = []
  for key in Inputs.keys():
    times.append(0.0)
    if key in map.keys():
      times.append(Inputs[key][0])
  times = sorted(times, key=float)

  outcome=[]
  dictOUTpre={}
  for time in times:
    inputToPass={}
    dictOUT = self.runStatic(container, inputToPass)
    if not dictOUTpre:
      pass
      #for var in 
      #dictOUTpre = copy.deepcopy(dictOUT)
  '''

def findAllRecursive(node, element):
  """
    A function for recursively traversing a node in an elementTree to find
    all instances of a tag.
    Note that this method differs from findall() since it goes for all nodes,
    subnodes, subsubnodes etc. recursively
    @ In, node, ET.Element, the current node to search under
    @ In, element, str, the string name of the tags to locate
    @ InOut, result, list, a list of the currently recovered results
  """
  result=[]
  for elem in node.iter(tag=element):
    result.append(elem)
  return result


