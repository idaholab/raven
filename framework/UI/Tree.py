#!/usr/bin/env python

class Node(object):
  def __init__(self, _id, parent=None, level=0):
    self.id = _id
    if parent is None:
      self.parent = self
    else:
      self.parent = parent
    self.level = level
    self.children = []

  def addChild(self,_id,level=0):
    self.children.append(Node(_id,self,level))

  def getNode(self,_id):
    if _id == self.id:
      return self
    else:
      for child in self.children:
        node = child.getNode(_id)
        if node is not None:
          return node
      return None

  def getLeafCount(self):
    if len(self.children) == 0:
      return 1
    count = 0
    for child in self.children:
      count += child.getLeafCount()
    return count

  def Layout(self,xoffset,width):
    ids = [self.id]
    points = [(xoffset+width/2.,self.level)]
    edges = []
    if len(self.children) > 0:
      totalCount = self.getLeafCount()
      myOffset = xoffset

      def cmp(a,b):
        if a.level > b.level:
          return -1
        return 1

      children = sorted(self.children, cmp=cmp)
      for child in children:
        edges.append((self.id,child.id))

        count = child.getLeafCount()
        myWidth = float(count)/totalCount*width
        (childIds,childPoints,childEdges) = child.Layout(myOffset,myWidth)
        ids.extend(childIds)
        points.extend(childPoints)
        edges.extend(childEdges)
        myOffset += myWidth
    return (ids,points,edges)