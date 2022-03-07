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
Created on Oct 27, 2016

@author: alfoa
"""
#----- python 2 - 3 compatibility
from __future__ import division, print_function, absolute_import
#----- end python 2 - 3 compatibility
#External Modules------------------------------------------------------------------------------------
import sys
import itertools
import copy
#External Modules End--------------------------------------------------------------------------------
#Internal Modules------------------------------------------------------------------------------------
from . import utils
#Internal Modules End--------------------------------------------------------------------------------


class graphObject(object):
  """
    This is a class that crates a graph object.
    It is inspired from the webisite http://www.python-course.eu/graphs_python.php
  """
  def __init__(self, graphDict=None):
    """
      Initializes a graph object
      If no dictionary or None is given, an empty dictionary will be used
      @ In, graphDict, dict, the graph dictionary ({'Node':[connectedNode1,connectedNode2, etc.]} where connectedNode is a node that relies on Node
    """
    if graphDict is None:
      graphDict = {}
    self.__graphDict = { k.strip():v for k, v in graphDict.items()}

  def vertices(self):
    """
      Returns the vertices of a graph
      @ In, None
      @ Out, vertices, list, list of vertices
    """
    return list(self.__graphDict.keys())

  def edges(self):
    """
      This method returns the edges of the graph
      Edges are represented as sets with one (a loop
      back to the vertex) or two vertices
      @ In, None
      @ Out, edges, list, list of edges ([set(vertex,neighboor)]
    """
    return self.__generateEdges()

  def addVertex(self, vertex):
    """
      If the vertex "vertex" is not in
      self.__graphDict, a key "vertex" with an empty
      list as a value is added to the dictionary.
      Otherwise nothing has to be done.
      @ In, vertex, string, the new vertex (e.g. 'a')
      @ Out, None
    """
    if vertex not in self.__graphDict:
      self.__graphDict[vertex] = []

  def addEdge(self, edge):
    """
      Method to add edges
      It assumes that edge is of type set, tuple or list;
      between two vertices can be multiple edges
      @ In, edge, set, the new edge
      @ Out, None
    """
    edge = set(edge)
    vertex1 = edge.pop()
    if edge:
      # not a loop
      vertex2 = edge.pop()
    else:
      # a loop
      vertex2 = vertex1
    if vertex1 in self.__graphDict:
      self.__graphDict[vertex1].append(vertex2)
    else:
      self.__graphDict[vertex1] = [vertex2]

  def __generateEdges(self):
    """
      A method generating the edges of the
      graph "graph". Edges are represented as sets
      with one (a loop back to the vertex) or two
      vertices
      @ In, None
      @ Out, edges, list, list of edges ([set(vertex,neighboor)]
    """
    edges = []
    for vertex in self.__graphDict:
      for neighbour in self.__graphDict[vertex]:
        if {neighbour, vertex} not in edges:
          edges.append({vertex, neighbour})
    return edges

#   def __str__(self):
#       res = "vertices: "
#       for k in self.__graphDict:
#           res += str(k) + " "
#       res += "\nedges: "
#       for edge in self.__generateEdges():
#           res += str(edge) + " "
#       return res

  def findIsolatedVertices(self):
    """
      This method ispects the graph and returns a list of isolated vertices.
      WARNING: the self.__extendDictForGraphTheory() is used here, and never store the outputs
      of this method into the self.__graphDict.
      @ In, None
      @ Out, isolated, list, list of isolated nodes (verteces)
    """
    graph = self.__extendDictForGraphTheory()
    isolated = []
    for vertex in graph:
      if not graph[vertex]:
        isolated += [vertex]
    return isolated

  def findPath(self, startVertex, endVertex, path=[]):
    """
      Method to find a path from startVertex to endVertex in graph
      @ In, startVertex, string, the starting vertex
      @ In, endVertex, string, the ending vertex
      @ Out, extendedPath, list, list of verteces (path)
    """
    graph = self.__graphDict
    path = path + [startVertex]
    extendedPath = None
    if startVertex == endVertex:
      extendedPath = path
    elif startVertex not in graph:
      extendedPath = None
    else:
      for vertex in graph[startVertex]:
        if vertex not in path:
          extendedPath = self.findPath(vertex, endVertex, path)
          if extendedPath:
            return extendedPath
    return extendedPath

  def isALoop(self):
    """
      Method to check if the graph has some loops
      Return True if the directed graph self.__graphDict has a cycle.
      @ In, None
      @ Out, isALoop, bool, is a loop?
    """
    path = set()
    g    = self.__graphDict

    def visit(vertex):
      """
        Method to visit the neighbour of a vertex
        @ In, vertex, string, the vertex
        @ Out, visit, bool, has the vertex a neighbour?
      """
      path.add(vertex)
      for neighbour in g.get(vertex, ()):
        if neighbour in path or visit(neighbour):
          return True
      path.remove(vertex)
      return False
    return any(visit(v) for v in g)

  def findAllPaths(self, startVertex, endVertex, path=[]):
    """
      Method to find all paths from startVertex to
      endVertex in graph
      @ In, startVertex, string, the starting vertex
      @ In, endVertex, string, the ending vertex
      @ Out, paths, list, list of list of verteces (paths)
    """
    graph = self.__graphDict
    path = path + [startVertex]
    if startVertex == endVertex:
      return [path]
    if startVertex not in graph:
      return []
    paths = []
    for vertex in graph[startVertex]:
      if vertex not in path:
        extendedPaths = self.findAllPaths(vertex,endVertex,path)
        for p in extendedPaths:
          paths.append(p)
    return paths

  def isConnected(self, graphDict, verticesEncountered = None, startVertex=None):
    """
      Method that determines if the graph is connected (graph theory connectivity)
      @ In, verticesEncountered, set, the encountered vertices (generally it is None when called from outside)
      @ In, graphDict, dict, the graph dictionary
      @ In, startVertex, string, the starting vertex
      @ Out, isConnected, bool, is the graph fully connected?
    """
    if verticesEncountered is None:
      verticesEncountered = set()
    vertices = list(graphDict.keys())
    if not startVertex:
      # chosse a vertex from graph as a starting point
      startVertex = vertices[0]
    verticesEncountered.add(startVertex)
    if len(verticesEncountered) != len(vertices):
      for vertex in graphDict[startVertex]:
        if vertex not in verticesEncountered:
          if self.isConnected(graphDict, verticesEncountered, vertex):
            return True
    else:
      return True
    return False

  def isConnectedNet(self):
    """
      Method that determines if the graph is a connected net (all the vertices are connect
      with each other through other vertices).
      WARNING: the self.__extendDictForGraphTheory() is used here, and never store the outputs
      of this method into the self.__graphDict.
      @ In, None
      @ Out, graphNetConnected, bool, is the graph net fully connected?
    """
    graphDict = self.__extendDictForGraphTheory()
    graphNetConnected = self.isConnected(graphDict)
    return graphNetConnected

  def __extendDictForGraphTheory(self):
    """
      Method to extend the graph dictionary in order to be accepted by the graph theory.
      WARNING: This is method is only used to extend the __graphDict, and should be only
      used for isConnectedNet method. The extended dictionary should not be stored in
      __graphDict. This is because the class graphObject is supposed to work with
      directed graph to determine the execution orders of the Models listed under EnsembleModel.
      @ In, None
      @ Out, graphDict, dict, the extended graph dictionary, used for isConnectedNet and findIsolatedVertices
    """
    graphDict = copy.deepcopy(self.__graphDict)
    # Extend graph dictionary generated by ensemble model to graph theory acceptable dictionary
    # Basicly, if a -> b (a is connected to b), the self.__graphDict = {'a':['b'], 'b':[]}
    # Since a is connected to b, from the graph theory, the dictionary should become {'a':['b'], 'b':['a']}
    for inModel, outModelList in self.__graphDict.items():
      for outModel in outModelList:
        if inModel not in graphDict[outModel]:
          graphDict[outModel].append(inModel)
    return graphDict

  def findAllUniquePaths(self, startVertices=None):
    """
      This method finds all the unique paths in the graph
      N.B. This method is not super efficient but since it is used generally at construction stage
      of the graph, it is not a big problem
      @ In, startVertices, list, optional, list of start vertices
      @ Out, uniquePaths,list, list of unique paths (verteces)
    """
    paths = []
    if not startVertices:
      startVertices = self.vertices()
    for vert in startVertices:
      for vertex in self.vertices():
        if vertex != vert:
          paths.extend(self.findAllPaths(vert, vertex))
    uniquePaths = list(utils.filterAllSubSets(paths))
    return uniquePaths

  def createSingleListOfVertices(self,uniquePaths=None):
    """
      This method is aimed to create a list of vertices from all the unique
      paths present in this graph
      @ In, uniquePaths, list of list, optional, list of unique paths. If not present, the unique paths are going to be determined
      @ Out, singleList, list, list of vertices in "ascending" order
    """
    if uniquePaths is None:
      uniquePaths = self.findAllUniquePaths()
    singleList = []
    for pathCnt in range(len(uniquePaths)):
      singleList = utils.mergeSequences(singleList,uniquePaths[pathCnt])
    return singleList

  def vertexDegree(self, vertex):
    """
      Method to get the degree of a vertex is the number of edges connecting
      it (i.e. the number of adjacent vertices). Loops are counted double (i.e. every
      occurence of vertex in the list
      of adjacent vertices.
      @ In, vertex, string, the vertex whose degree needs to be reported
      @ Out, degree, int, the degree of the vertex
    """
    adjVertices =  self.__graphDict[vertex]
    degree = len(adjVertices) + adjVertices.count(vertex)
    return degree

  def degreeSequence(self):
    """
      Method to calculate the degree sequence
      @ In, None
      @ Out, seq, tuple, the degree sequence
    """
    seq = []
    for vertex in self.__graphDict:
      seq.append(self.vertexDegree(vertex))
    seq.sort(reverse=True)
    return tuple(seq)

  @staticmethod
  def isDegreeSequence(sequence):
    """
      Method returns True, if the sequence "sequence" is a
      degree sequence, i.e. a non-increasing sequence.
      Otherwise False is returned.
      @ In, sequence, list, list of integer sequence
      @ Out, isDegreeSeq, bool, is a degree sequence? (i.e. the sequence sequence is non-increasing)
    """
    # check if the sequence sequence is non-increasing:
    isDegreeSeq = all( x>=y for x, y in zip(sequence, sequence[1:]))
    return isDegreeSeq

  def minDelta(self):
    """
      Method to compute the minimum degree of the vertices
      @ In, None
      @ Out, minDegree, integer, the minimum degree of the vertices
    """
    minDegree = 2**62 #sys.maxint
    for vertex in self.__graphDict:
      vertexDegree = self.vertexDegree(vertex)
      if vertexDegree < minDegree:
        minDegree = vertexDegree
    return minDegree

  def maxDelta(self):
    """
      Method to compute the maximum degree of the vertices
      @ In, None
      @ Out, maxDegree, integer, the minimum degree of the vertices
    """
    maxDegree = 0
    for vertex in self.__graphDict:
      vertexDegree = self.vertexDegree(vertex)
      if vertexDegree > maxDegree:
        maxDegree = vertexDegree
    return maxDegree

  def density(self):
    """
      Method to compute the density of the graph
      @ In, None
      @ Out, dens, integer, density of a graph
    """
    V = len(self.__graphDict.keys())
    E = len(self.edges())
    try:
      dens = 2.0 * E / (V *(V - 1))
    except:
      dens = 0
    return dens

  def diameter(self):
    """
      Method to compute the diameter of the graph
      @ In, None
      @ Out, diameter, integer, ensity of a graph
    """
    v    = self.vertices()
    vrev = list(reversed(v))
    pairs = list(itertools.combinations(v,2))
    pairs.extend(list(itertools.combinations(vrev,2)))
    smallestPaths = []
    for (s,e) in pairs:
      paths = self.findAllPaths(s,e)
      sortedPath = sorted(paths, key=len)
      smallest = sortedPath[0] if len(sortedPath) > 0 else None
      if smallest is not None:
        smallestPaths.append(smallest)
    smallestPaths.sort(key=len)
    # longest path is at the end of list,
    # i.e. diameter corresponds to the length of this path
    diameter = len(smallestPaths[-1])
    return diameter

  @staticmethod
  def erdoesGallai(sequence):
    """
      Static method to check if the condition of the Erdoes-Gallai inequality is fullfilled
      @ In, sequence, list, list of integer representing the sequence
      @ Out, metConditions, bool, True if the Erdoes-Gallai inequality is fullfilled
    """
    metConditions = True
    if sum(sequence) % 2:
      # sum of sequence is odd
      metConditions = False
    else:
      if Graph.isDegreeSequence(sequence):
        for k in range(1,len(sequence) + 1):
          left = sum(sequence[:k])
          right =  k * (k-1) + sum([min(x,k) for x in sequence[k:]])
          if left > right:
            metConditions = False
            break
      else:
        # sequence is increasing
        metConditions = False
    return metConditions
