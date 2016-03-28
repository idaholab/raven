"""
Created on Jan 28, 2014
TreeStructure. 2 classes Node, NodeTree
@author: alfoa
"""

#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

class Node(object):
  """
    The Node class. It represents the base for each TreeStructure construction
  """
  def __init__(self, name, valuesIn={}, text=''):
    """
      Initialize Tree,
      @ In, name, string, is the node name
      @ In, valuesIn, dict, optional, is a dictionary of values
      @ In, text, string, optional, the node's text, as <name>text</name>
    """
    values         = valuesIn.copy()
    self.name      = name
    self.values    = values
    self.text      = text
    self._branches = []
    self.parentname= None
    self.parent    = None
    self.depth     = 0

  def __repr__(self):
    """
      Overload the representation of this object... We want to show the name and the number of branches!!!!
      @ In, None
      @ Out, __repr__, string, the representation of this object
    """
    return "<Node %s at 0x%x containing %s branches>" % (repr(self.name), id(self), repr(len(self._branches)))

  def copyNode(self):
    """
      Method to copy this node and return it
      @ In, None
      @ Out, node, instance, a new instance of this node
    """
    node = self.__class__(self.name, self.values)
    node[:] = self
    return node

  def isAnActualBranch(self,branchName):
    """
      Method to check if branchName is an actual branch
      @ In, branchName, string, the branch name
      @ Out, isHere, bool, True if it found
    """
    isHere = False
    for branchv in self._branches:
      if branchName.strip() == branchv.name: isHere = True
    return isHere

  def numberBranches(self):
    """
      Method to get the number of branches
      @ In, None
      @ Out, len, int, number of branches
    """
    return len(self._branches)

  def appendBranch(self, node):
    """
      Method used to append a new branch to this node
      @ In, node, Node, the newer node
      @ Out, None
    """
    node.parentname = self.name
    node.parent     = self
    node.updateDepth()
    self._branches.append(node)

  def updateDepth(self):
    """
      updates the 'depth' parameter throughout the tree
      @In, None
      @Out, None
    """
    if self.parent=='root': self.depth=0
    else: self.depth = self.parent.depth+1
    for node in self._branches:
      node.updateDepth()

  def extendBranch(self, nodes):
    """
      Method used to append subnodes from a sequence of them
      @ In, nodes, list, list of NodeTree
      @ Out, None
    """
    for nod in nodes:
      nod.parentname = self.name
      nod.parent     = self
    self._branches.extend(nodes)

  def insertBranch(self, pos, node):
    """
      Method used to insert a new branch in a given position
      @ In, pos, integer, the position
      @ In, node, Node, the newer node
      @ Out, None
    """
    node.parentname = self.name
    node.parent     = self
    self._branches.insert(pos, node)

  def removeBranch(self, node):
    """
      Method used to remove a subnode
      @ In, node, Node, the node to remove
      @ Out, None
    """
    self._branches.remove(node)

  def findBranch(self, path):
    """
      Method used to find the first matching branch (subnode)
      @ In, path, string, is the name of the branch or the path
      @ Out, node, Node, the matching subnode
    """
    return NodePath().find(self, path)

  def findallBranch(self, path):
    """
      Method used to find all the matching branches (subnodes)
      @ In, path, string, is the name of the branch or the path
      @ Out, nodes, list, list of all the matching subnodes
    """
    return NodePath().findall(self, path)

  def iterfind(self, path):
    """
      Method used to find all the matching branches (subnodes)
      @ In, path, string, is the name of the branch or the path
      @ Out, iterator, iterator instance, iterator containing all matching nodes
    """
    return NodePath().iterfind(self, path)

  def getParentName(self):
    """
      Method used to get the parentname
      @ In, None
      @ Out, parentName, string, the parent name
    """
    return self.parentname

  def clearBranch(self):
    """
      Method used clear this node
      @ In, None
      @ Out, None
    """
    self.values.clear()
    self._branches = []

  def get(self, key, default=None):
    """
      Method to get a value from this element tree
      If the key is not present, None is returned
      @ In, key, string, id name of this value
      @ In, default, object, optional, an optional default value returned if not found
      @ Out, object, object, the coresponding value or default
    """
    return self.values.get(key, default)

  def add(self, key, value):
    """
      Method to add a new value into this node
      If the key is already present, the corresponding value gets updated
      @ In, key, string, id name of this value
      @ In, value, whatever type, the newer value
    """
    self.values[key] = value

  def keys(self):
    """
      Method to return the keys of the values dictionary
      @ In, None
      @ Out, keys, list, the values keys
    """
    return self.values.keys()

  def getValues(self):
    """
      Method to return values dictionary
      @ In, None
      @ Out, self.values, dict, the values
    """
    return self.values

  def iter(self, name=None):
    """
      Creates a tree iterator.  The iterator loops over this node
      and all subnodes and returns all nodes with a matching name.
      @ In, name, string, optional, name of the branch wanted
      @ Out, e, iterator, the iterator
    """
    if name == "*":
      name = None
    if name is None or self.name == name:
      yield self
    for e in self._branches:
      for e in e.iter(name):
        yield e

  def iterProvidedFunction(self, providedFunction):
    """
      Creates a tree iterator.  The iterator loops over this node
      and all subnodes and returns all nodes for which the providedFunction returns True
      @ In, providedFunction, method, the function instance
      @ Out, e, iterator, the iterator
    """
    if  providedFunction(self.values):
      yield self
    for e in self._branches:
      for e in e.iterProvidedFunction(providedFunction):
        yield e

  def iterEnding(self):
    """
      Creates a tree iterator for ending branches.  The iterator loops over this node
      and all subnodes and returns all nodes without branches
      @ In, None
      @ Out, e, iterator, the iterator
    """
    if len(self._branches) == 0:
      yield self
    for e in self._branches:
      for e in e.iterEnding():
        yield e

  def iterWholeBackTrace(self,startnode):
    """
      Method for creating a sorted list (backward) of nodes starting from node named "name"
      @ In, startnode, Node, the node
      @ Out, result, list, the list of nodes
    """
    result    =  []
    parent    =  startnode.parent
    ego       =  startnode
    while parent:
      result.insert (0, ego)
      parent, ego  =  parent.parent, parent
    if ego.parentname == 'root': result.insert (0, ego)
    return result

  def setText(self,entry):
    """
      Sets the text of the node, as <node>text</node>.
      @ In, entry, string, string to store as node text
      @ Out, None
    """
    self.text = str(entry)

  def writeNode(self,dumpFileObj):
    """
      This method is used to write the content of the node into a file (it recorsevely prints all the sub-nodes and sub-sub-nodes, etc)
      @ In, dumpFileObj, file instance, file instance(opened file)
      @ Out, None
    """
    dumpFileObj.write(' '+'  '*self.depth + '<branch name="' + self.name + '" parent_name="' + self.parentname + '"'+ ' n_branches="'+str(self.numberBranches())+'" >\n')
    if len(self.values.keys()) >0: dumpFileObj.write(' '+'  '*self.depth +'  <attributes>\n')
    for key,value in self.values.items(): dumpFileObj.write(' '+'  '*self.depth+'    <'+ key +'>' + str(value) + '</'+key+'>\n')
    if len(self.values.keys()) >0: dumpFileObj.write(' '+'  '*self.depth +'  </attributes>\n')
    for e in self._branches: e.writeNode(dumpFileObj)
    if self.numberBranches()>0: dumpFileObj.write(' '+'  '*self.depth + '</branch>\n')

  def stringNode(self,msg):
    """
      As writeNode, but returns a string representation of the tree instead of writing it to file.
      @ In, msg, string, the string to populate
      @ Out, msg, string, the modified string
    """
    msg+=''+'  '*self.depth + '<' + self.name + '>'+self.text
    if self.numberBranches()==0:msg+='</'+self.name+'>'
    msg+='\n'
    if len(self.values.keys()) >0: msg+=''+'  '*self.depth +'  <attributes>\n'
    for key,value in self.values.items(): msg+=' '+'  '*self.depth+'    <'+ key +'>' + str(value) + '</'+key+'>\n'
    if len(self.values.keys()) >0: msg+=''+'  '*self.depth +'  </attributes>\n'
    for e in self._branches: msg=e.stringNode(msg)
    if self.numberBranches()>0: msg+=''+'  '*self.depth + '</'+self.name+'>\n'
    return msg

#################
#   NODE TREE   #
#################
class NodeTree(object):
  """
    NodeTree class. The class tha realizes the Tree Structure
  """
  def __init__(self, node=None):
    """
      Constructor
      @ In, node, Node, optional, the rootnode
      @ Out, None
    """
    self._rootnode = node
    if node: node.parentname='root'

  def getrootnode(self):
    """
      Get the root node reference
      @ In, None
      @ Out, self._rootnode, Node, the root node
    """
    return self._rootnode

  def _setrootnode(self, node):
    """
      Method used to replace the rootnode with this node
      @ In, node, Node, the newer node
      @ Out, None
    """
    self._rootnode = node

  def updateNodeName(self,path, newName):
    """
      Method to update the name of a node
      @ In, path, string, the node name or full path
      @ In, newName, string, the new name
      @ Out, None
    """
    if path == "root": node = self.getrootnode()
    else             : node = self.find(path)
    if node != None: node.name = newName

  def iter(self, name=None):
    """
      Method for creating a tree iterator for the root node
      @ In, name, string, the path or the node name
      @ Out, iter, iterator, the iterator
    """
    if name == 'root': return self.__rootnode
    else:              return self._rootnode.iter(name)

  def iterEnding(self):
    """
      Method for creating a tree iterator for the root node (ending branches)
      @ In, None
      @ Out, iterEnding, iterator, the iterator
    """
    return self._rootnode.iterEnding()

  def iterProvidedFunction(self, providedFunction):
    """
      Method for creating a tree iterator for the root node (depending on returning of provided function)
      @ In, providedFunction, instance, the function
      @ Out, iterProvidedFunction, iterator, the iterator
    """
    return self._rootnode.iterProvidedFunction(providedFunction)

  def iterWholeBackTrace(self,startnode):
    """
      Method for creating a sorted list (backward) of nodes starting from node named "name"
      @ In, startnode, Node, the node
      @ Out, iterWholeBackTrace, list, the list of pointers to nodes
    """
    return self._rootnode.iterWholeBackTrace(startnode)

  def find(self, path):
    """
      Method to find the first toplevel node with a given name
      @ In, path, string, the path or name
      @ Out, node, Node, first matching node or None if no node was found
    """
    if self._rootnode.name == path: return self.getrootnode()
    if path[:1] == "/":
      path = "." + path
    return self._rootnode.findBranch(path)

  def findall(self, path):
    """
      Method to find the all toplevel nodes with a given name
      @ In, path, string, the path or name
      @ Out, findall, list of Node iterators, A list or iterator containing all matching nodes
    """
    if self._rootnode.name == path: return [self.getrootnode()]
    if path[:1] == "/":
      path = "." + path
    return self._rootnode.findallBranch(path)

  def iterfind(self, path):
    """
      Method to find the all matching subnodes with a given name
      @ In, path, string, the path or name
      @ Out, iterfind, list of Node iterators, a sequence of node instances
    """
    if path[:1] == "/":
      path = "." + path
    return self._rootnode.iterfind(path)

  def writeNodeTree(self,dumpFile):
    """
      This method is used to write the content of the whole tree into a file
      @ In, dumpFile, file instance or string, filename (string) or file instance(opened file)
      @ Out, None
    """
    if type(dumpFile).__name__ == 'FileObject' : myFile = open(dumpFile,'w')
    else                                       : myFile = dumpFile
    myFile.write('<NodeTree name = "'+self._rootnode.name+'">\n')
    self._rootnode.writeNode(myFile)
    myFile.write('</NodeTree>\n')
    if type(dumpFile).__name__ == 'FileObject' : myFile.close()

  def stringNodeTree(self,msg=''):
    """
      As writeNodeTree, but creates a string representation instead of writing to a file.
      @ In, msg, string, the string to populate
      @ Out, msg, string, the populated string
    """
    msg=str(msg)
    msg=self._rootnode.stringNode(msg)
    return msg

####################
#  NodePath Class  #
#  used to iterate #
####################
class NodePath(object):
  """
    NodePath class. It is used to perform iterations over the Tree
  """
  def find(self, node, name):
    """
      Method to find a matching node
      @ In, node, Node, the node (Tree) where the 'name' node needs to be found
      @ In, name, string, the name of the node that needs to be found
      @ Out, nod, Node, the matching node (if found) else None
    """
    for nod in node._branches:
      if nod.name == name:
        return nod
    return None

  def iterfind(self, node, name):
    """
      Method to create an iterator starting from a matching node
      @ In, node, Node, the node (Tree) where the 'name' node needs to be found
      @ In, name, string, the name of the node from which the iterator needs to be created
      @ Out, nod, Node iterator, the matching node (if found) else None
    """
    if name[:3] == ".//":
      for nod in node.iter(name[3:]):
        yield nod
      for nod in node:
        if nod.name == name:
          yield nod

  def findall(self, node, name):
    """
      Method to create an iterator starting from a matching node for all the nodes
      @ In, node, Node, the node (Tree) where the 'name' node needs to be found
      @ In, name, string, the name of the node from which the iterator needs to be created
      @ Out, nodes, list, list of all matching nodes
    """
    nodes = list(self.iterfind(node, name))
    return nodes

def isnode(node):
  """
    Method to create an iterator starting from a matching node for all the nodes
    @ In, node, object, the node that needs to be checked
    @ Out, isinstance, bool, is a node instance?
  """
  return isinstance(node, Node) or hasattr(node, "name")
