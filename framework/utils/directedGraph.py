"""
Created on Oct 27, 2016

@author: alfoa
"""
#----- python 2 - 3 compatibility
from __future__ import division, print_function, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#----- end python 2 - 3 compatibility
#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

class graphObject(object):
  """
    This is a class that crates a graph object.
    It is inspired from the webisite http://www.python-course.eu/graphs_python.php
  """
  def __init__(self, graphDict=None):
    """
      Initializes a graph object
      If no dictionary or None is given, an empty dictionary will be used
      @ In, graphDict, dict, the graph dictionary ({'Node':[connectedNode1,connectedNode2, etc.]}
    """
    if graphDict == None: graphDict = {}
    self.__graphDict = graphDict

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
    if vertex not in self.__graphDict: self.__graphDict[vertex] = []

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
      A static method generating the edges of the
      graph "graph". Edges are represented as sets
      with one (a loop back to the vertex) or two
      vertices
      @ In, None
      @ Out, edges, list, list of edges ([set(vertex,neighboor)]
    """
    edges = []
    for vertex in self.__graphDict:
      for neighbour in self.__graphDict[vertex]:
        if {neighbour, vertex} not in edges:edges.append({vertex, neighbour})
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
      @ In, None
      @ Out, isolated, list, list of isolated nodes (verteces)
    """
    graph = self.__graphDict
    isolated = []
    for vertex in graph:
      print(isolated, vertex)
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
          if extendedPath: return extendedPath
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
        if neighbour in path or visit(neighbour): return True
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
    if startVertex == endVertex: return [path]
    if startVertex not in graph: return []
    paths = []
    for vertex in graph[startVertex]:
      if vertex not in path:
        extendedPaths = self.findAllPaths(vertex,endVertex,path)
        for p in extendedPaths: paths.append(p)
    return paths

  def isConnected(self, verticesEncountered = None, startVertex=None):
    """
      Method that determines if the graph is connected
      @ In, verticesEncountered, set, the encountered vertices (generally it is None when called from outside)
      @ In, startVertex, string, the starting vertex
      @ Out, isConnected, bool, is the graph fully connected?
    """
    if verticesEncountered is None: verticesEncountered = set()
    gdict = self.__graphDict
    vertices = gdict.keys()
    if not startVertex:
      # chosse a vertex from graph as a starting point
      startVertex = vertices[0]
    verticesEncountered.add(startVertex)
    if len(verticesEncountered) != len(vertices):
      for vertex in gdict[startVertex]:
        if vertex not in verticesEncountered:
          if self.isConnected(verticesEncountered, vertex): return True
    else: return True
    return False

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
      @ Out, isDegreeSequence, bool, is a degree sequence? (i.e. the sequence sequence is non-increasing)
    """
    # check if the sequence sequence is non-increasing:
    return all( x>=y for x, y in zip(sequence, sequence[1:]))

  def minDelta(self):
    """
      Method to compute the minimum degree of the vertices
      @ In, None
      @ Out, minDegree, integer, the minimum degree of the vertices
    """
    minDegree = 100000000
    for vertex in self.__graphDict:
      vertexDegree = self.vertexDegree(vertex)
      if vertexDegree < minDegree: minDegree = vertexDegree
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
      if vertexDegree > maxDegree: maxDegree = vertexDegree
    return maxDegree

  def density(self):
    """
      Method to compute the density of the graph
      @ In, None
      @ Out, density, integer, ensity of a graph
    """
    V = len(self.__graphDict.keys())
    E = len(self.edges())
    return 2.0 * E / (V *(V - 1))

  def diameter(self):
    """
      Method to compute the diameter of the graph
      @ In, None
      @ Out, diameter, integer, ensity of a graph
    """
    v = self.vertices()
    pairs = [ (v[i],v[j]) for i in range(len(v)-1) for j in range(i+1, len(v))]
    smallestPaths = []
    for (s,e) in pairs:
      paths = self.findAllPaths(s,e)
      smallest = sorted(paths, key=len)[0]
      smallestPaths.append(smallest)
    smallestPaths.sort(key=len)
    # longest path is at the end of list,
    # i.e. diameter corresponds to the length of this path
    diameter = len(smallestPaths[-1])
    return diameter

  @staticmethod
  def erdoesGallai(sequence):
    """
      Static methoed to check if the condition of the Erdoes-Gallai inequality is fullfilled
      @ In, sequence, list, list of integer representing the sequence
      @ Out, erdoesGallai, bool, True if the Erdoes-Gallai inequality is fullfilled
    """
    if sum(sequence) % 2:
      # sum of sequence is odd
      return False
    if Graph.isDegreeSequence(sequence):
      for k in range(1,len(sequence) + 1):
        left = sum(sequence[:k])
        right =  k * (k-1) + sum([min(x,k) for x in sequence[k:]])
        if left > right:return False
    else:
      # sequence is increasing
      return False
    return True
