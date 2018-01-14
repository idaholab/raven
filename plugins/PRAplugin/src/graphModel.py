from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
import math
import xml.etree.ElementTree as ET
from utils import utils
from utils import graphStructure as GS
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
    print('here_xml')

    container.modelFile = None
    container.nodesIN   = None
    container.nodesOUT  = None

    for child in xmlNode:
      if   child.tag == 'nodesIN':
        container.nodesIN  = [var.strip() for var in child.text.split(",")]
      elif child.tag == 'nodesOUT':
        container.nodesOUT = [var.strip() for var in child.text.split(",")]
      elif child.tag == 'modelFile':
        container.modelFile = child.text + '.xml'

    if container.nodesIN is None:
      print('nodesIN Error')
    if container.nodesOUT is None:
      print('nodesOUT Error')
    if container.modelFile is None:
      print('modelFile Error')


  def initialize(self, container,runInfoDict,inputFiles):
    """
      Method to initialize this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """
    print('here_init')

    container.nodes = {}  
    container.deg   = {}

    container.runInfo = runInfoDict
    print(container.modelFile)
    self.createGraph(container,container.modelFile)


  def createGraph(self,container,file):
    print('here_create')
    graph = ET.parse(container.runInfo['WorkingDir'] + '/' + file)
    graph = findAllRecursive(graph,'Graph')

    for node in findAllRecursive(graph[0], 'node'):
      nodeName = node.get('name')
      nodeChilds = None
      deg = None
      for child in node:
        if child.tag == 'childs':
          nodeChilds = [var.strip() for var in child.text.split(",")]
        if child.tag == 'deg':
          deg = float(child.text)
      container.nodes[nodeName]=nodeChilds
      container.deg[nodeName]=deg

    ravenGraph = GS.graphObject(container.nodes)

    container.pathDict = {}
    for nodeO in container.nodesOUT:
      paths = []
      for nodeI in container.nodesIN:
        paths = paths + ravenGraph.findAllPaths(nodeI,nodeO)
      container.pathDict[nodeO] = paths

  def run(self, container, Inputs):
    """
      This is a simple example of the run method in a plugin.
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN

    """
    print('here_run')
    for key in Inputs:
      if key in self.nodes.keys():
        if Inputs[key] == 0.0:       # 1=operating ; 0=failed
          for nodeO in self.pathDict.keys():
            for path in self.pathDict[nodeO]:
              if key in path:
                self.pathDict[nodeO].remove(path)
        else: 
          print('invalid value')

    for nodeO in self.pathDict.keys():
      if self.pathDict[nodeO]:
        containter[nodeO] = 1.
      else:
        containter[nodeO] = 0.
    
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


