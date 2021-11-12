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
    This file contains a Node data structure which is used to construct a Tree.
    This tree will handle performing layout for UI purposes.
"""

#For future compatibility with Python 3
from __future__ import division, print_function, absolute_import
#End compatibility block for Python 3

import numpy as np

class Node(object):
  """
    A Node data structure for use in a larger Tree.
  """
  def __init__(self, _id, parent=None, level=0, size=1):
    """
      THe Initialization method that will create a leaf node.
      @ In, _id, object, unique identifier, usually an integer, but the data
        type is not strictly enforced and can be anything as long as the rest
        of the tree uses it consistently.
      @ In, parent, object, a unique identifier for another node that is to be
        this node's parent.
      @ In, level, float, the vertical level in the tree where this node will
        appear.
      @ In, size, int, the number of items associated to this node.
      @ Out
    """
    self.id = _id
    if parent is None:
      self.parent = self
    else:
      self.parent = parent
    self.level = level
    self.size = size
    self.children = []

  def addChild(self, _id, level=0, size=1):
    """
      A method to add a child node to this node.
      @ In, _id, object, a new unique identifier for the child.
      @ In, level, float, the vertical level in the tree where the new child
        will appear.
      @ In, size, int, the number of items associated to the new node.
      @ Out, node, the new node.
    """
    node = Node(_id,self,level,size)
    self.children.append(node)
    return node

  def getNode(self, _id):
    """
      Method to find a node in the tree given an identifier.
      @ In, _id, object, node you are looking for.
      @ Out, node, pointer to the actual node.
    """
    if _id == self.id:
      return self
    else:
      for child in self.children:
        node = child.getNode(_id)
        if node is not None:
          return node
      return None

  def getLeafCount(self, truncationSize=0, truncationLevel=0):
    """
      Get the leaf count of the current node restricted to two truncation
      parameters.
      @ In, truncationSize, int, a minimum size for the maximum child.
      @ In, truncationLevel, float, a minimum vertical level for the maximum
        child.
      @ Out, count, int, the number of children of this node restricted to the
        two truncation parameters.
    """
    if len(self.children) == 0:
      return 1

    truncated = True

    for child in self.children:
      if child.level >= truncationLevel and child.size >= truncationSize:
        truncated = False

    count = 0
    for child in self.children:
      if not truncated:
        count += child.getLeafCount(truncationSize, truncationLevel)

    if count == 0:
      return 1

    return count

  def maximumChild(self, truncationSize=0, truncationLevel=0):
    """
      Get the maximum level of the current node restricted to two truncation
      parameters.
      @ In, truncationSize, int, a minimum size for the maximum child.
      @ In, truncationLevel, float, a minimum vertical level for the maximum
        child.
      @ Out, maxChild, Node, the maximum level child of this node.
    """
    maxChild = None

    truncated = True
    for child in self.children:
      if child.level >= truncationLevel and child.size >= truncationSize:
        truncated = False

    for child in self.children:
      if not truncated:
        if maxChild is None or maxChild.level < child.level:
          maxChild = child

    return maxChild

  def Layout(self,xoffset,width, truncationSize=0, truncationLevel=0):
    """
      Get the leaf count of the current node restricted to two truncation
      parameters.
      @ In, xoffset, float, a starting position for this node.
      @ In, width, float, the width reserved for this node.
      @ In, truncationSize, int, a minimum size for the maximum child.
      @ In, truncationLevel, float, a minimum vertical level for the maximum
        child.
      @ Out, (ids,points,edges), tuple where each of the outputs is described
        below:
          ids, list(object), the mapping of ids to array position.
          points, list((float,float)), the corresponding locations of the
            aforementioned ids list.
          edges, list((object,object))), a list of edges that connect one id
            to another, where the first item in the tuple is the id of the
            ancestor and the second is the id of the descendant.
    """
    ids = [self.id]
    points = [(xoffset+width/2.,self.level)]
    edges = []

    totalCount = self.getLeafCount(truncationSize,truncationLevel)
    if len(self.children) > 0 and totalCount > 1:

      myOffset = xoffset

      def key(a):
        """
          A function for getting a key for a Node
          @ In, a, Node, the node to get a key from.
          @ Out, value, the key
        """
        return a.level

      children = sorted(self.children, key=key)
      immediateDescendantXs = []
      truncated = True

      for child in children:
        if child.level >= truncationLevel and child.size >= truncationSize:
          truncated = False

      for child in children:
        if not truncated:
          edges.append((self.id,child.id))

          count = child.getLeafCount(truncationSize,truncationLevel)
          myWidth = float(count)/totalCount*width
          (childIds,childPoints,childEdges) = child.Layout(myOffset,myWidth,truncationSize,truncationLevel)
          ids.extend(childIds)
          points.extend(childPoints)
          edges.extend(childEdges)

          if len(childPoints) > 0:
            immediateDescendantXs.append(childPoints[0][0])
          myOffset += myWidth

      ## If this guy has children, then we will readjust its X location to be
      ## the average of its immediate descendants
      if len(immediateDescendantXs) > 0:
        points[0] = (np.average(immediateDescendantXs),self.level)
    return (ids,points,edges)
