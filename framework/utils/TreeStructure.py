'''
Created on Jan 28, 2014
TreeStructure. 2 classes Node, NodeTree
@author: alfoa
'''

#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
import utils
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

class Node(object):
  def __init__(self, name, valuesin={}, text=''):
    '''
      Initialize Tree,
      @ In, name, String, is the node name
      @ In, valuesin, is a dictionary of values
    '''
    values         = valuesin.copy()
    self.name      = name
    self.values    = values
    self.text      = text
    self._branches = []
    self.parentname= None
    self.parent    = None
    self.depth     = 0

  def __repr__(self):
    '''
      Overload the representation of this object... We want to show the name and the number of branches!!!!
    '''
    return "<Node %s at 0x%x containing %s branches>" % (repr(self.name), id(self), repr(len(self._branches)))

  def copyNode(self):
    '''
      Method to copy this node and return it
      @ In, None
      @ Out, a new instance of this node
    '''
    node = self.__class__(self.name, self.values)
    node[:] = self
    return node

  def isAnActualBranch(self,branchName):
    isHere = False
    for branchv in self._branches:
      if branchName.strip() == branchv.name: isHere = True
    return isHere

  def numberBranches(self):
    '''
      Method to get the number of branches
      @ In, None
      @ Out, int, number of branches
    '''
    return len(self._branches)

  def appendBranch(self, node):
    '''
      Method used to append a new branch to this node
      @ In, NodeTree, the newer node
    '''
    node.parentname = self.name
    node.parent     = self
    node.depth      = self.depth + 1
    self._branches.append(node)

  def extendBranch(self, nodes):
    '''
      Method used to append subnodes from a sequence of them
      @ In, list of NodeTree, nodes
    '''
    for nod in nodes:
      nod.parentname = self.name
      nod.parent     = self
    self._branches.extend(nodes)

  def insertBranch(self, pos, node):
    '''
      Method used to insert a new branch in a given position
      @ In, node, NodeTree, the newer node
      @ In, pos, integer, the position
    '''
    node.parentname = self.name
    node.parent     = self
    self._branches.insert(pos, node)

  def removeBranch(self, node):
    '''
      Method used to remove a subnode
      @ In, the node to remove
    '''
    self._branches.remove(node)

  def findBranch(self, path):
    '''
      Method used to find the first matching branch (subnode)
      @ In, path, string, is the name of the branch or the path
      @ Out, the matching subnode
    '''
    return NodePath().find(self, path)

  def findallBranch(self, path):
    '''
      Method used to find all the matching branches (subnodes)
      @ In, path, string, is the name of the branch or the path
      @ Out, all the matching subnodes
    '''
    return NodePath().findall(self, path)

  def iterfind(self, path):
    '''
      Method used to find all the matching branches (subnodes)
      @ In, path, string, is the name of the branch or the path
      @ Out, iterator containing all matching nodes
    '''
    return NodePath().iterfind(self, path)

  def getParentName(self):
    '''
      Method used to get the parentname
      @ In, None
      @ Out, parentName
    '''
    return self.parentname

  def clearBranch(self):
    '''
      Method used clear this node
      @ In, None
      @ Out, None
    '''
    self.values.clear()
    self._branches = []

  def get(self, key, default=None):
    '''
      Method to get a value from this element tree
      If the key is not present, None is returned
      @ In, key, string, id name of this value
      @ In, default, an optional default value returned if not found
      @ Out, the coresponding value or default
    '''
    return self.values.get(key, default)

  def add(self, key, value):
    '''
      Method to add a new value into this node
      If the key is already present, the corresponding value gets updated
      @ In, key, string, id name of this value
      @ In, value, whatever type, the newer value
    '''
    self.values[key] = value

  def keys(self):
    '''
      Method to return the keys of the values dictionary
      @ Out, the values keys
    '''
    return self.values.keys()

  def getValues(self):
    '''
      Method to return values dictionary
      @ Out, dict, the values
    '''
    return self.values

  def iter(self, name=None):
    '''
       Creates a tree iterator.  The iterator loops over this node
       and all subnodes and returns all nodes with a matching name.
       @ In, string, name of the branch wanted
    '''
    if name == "*":
      name = None
    if name is None or self.name == name:
      yield self
    for e in self._branches:
      for e in e.iter(name):
        yield e

  def iterProvidedFunction(self, providedFunction):
    '''
       Creates a tree iterator.  The iterator loops over this node
       and all subnodes and returns all nodes for which the providedFunction returns True
       @ In, string, name of the branch wanted
    '''
    if  providedFunction(self.values):
      yield self
    for e in self._branches:
      for e in e.iterProvidedFunction(providedFunction):
        yield e

  def iterEnding(self):
    '''
       Creates a tree iterator for ending branches.  The iterator loops over this node
       and all subnodes and returns all nodes without branches
    '''
    if len(self._branches) == 0:
      yield self
    for e in self._branches:
      for e in e.iterEnding():
        yield e

  def iterWholeBackTrace(self,startnode):
    '''
      Method for creating a sorted list (backward) of nodes starting from node named "name"
      @ In, startnode, Node, the node
      @ Out, the list
    '''
    result    =  []
    parent    =  startnode.parent
    ego       =  startnode
    while parent:
      result.insert (0, ego)
      parent, ego  =  parent.parent, parent
    if ego.parentname == 'root': result.insert (0, ego)
    return result

  def setText(self,entry):
    self.text = str(entry)

  def writeNode(self,dumpFileObj):
    '''
      This method is used to write the content of the node into a file (it recorsevely prints all the sub-nodes and sub-sub-nodes, etc)
      @ In, dumpFileObj, file instance, file instance(opened file)
    '''
    dumpFileObj.write(' '+'  '*self.depth + '<branch name="' + self.name + '" parent_name="' + self.parentname + '"'+ ' n_branches="'+str(self.numberBranches())+'" >\n')
    if len(self.values.keys()) >0: dumpFileObj.write(' '+'  '*self.depth +'  <attributes>\n')
    for key,value in self.values.items(): dumpFileObj.write(' '+'  '*self.depth+'    <'+ key +'>' + str(value) + '</'+key+'>\n')
    if len(self.values.keys()) >0: dumpFileObj.write(' '+'  '*self.depth +'  </attributes>\n')
    for e in self._branches: e.writeNode(dumpFileObj)
    if self.numberBranches()>0: dumpFileObj.write(' '+'  '*self.depth + '</branch>\n')

  def stringNode(self,msg):
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
  def __init__(self, node=None):
      self._rootnode = node
      if node: node.parentname='root'

  def getrootnode(self):
      return self._rootnode

  def _setrootnode(self, node):
    '''
      Method used to replace the rootnode with this node
      @ In, the newer node
    '''
    self._rootnode = node

  def iter(self, name=None):
    '''
      Method for creating a tree iterator for the root node
      @ In, name, string, the path or the node name
      @ Out, the iterator
    '''
    if name == 'root': return self.__rootnode
    else:              return self._rootnode.iter(name)

  def iterEnding(self):
    '''
      Method for creating a tree iterator for the root node (ending branches)
      @ Out, the iterator
    '''
    return self._rootnode.iterEnding()

  def iterProvidedFunction(self, providedFunction):
    '''
      Method for creating a tree iterator for the root node (depending on returning of provided function)
      @ Out, the iterator
    '''
    return self._rootnode.iterProvidedFunction(providedFunction)

  def iterWholeBackTrace(self,startnode):
    '''
      Method for creating a sorted list (backward) of nodes starting from node named "name"
      @ In, startnode, Node, the node
      @ Out, the list
    '''
    return self._rootnode.iterWholeBackTrace(startnode)

  def find(self, path):
    '''
      Method to find the first toplevel node with a given name
      @ In, path, string, the path or name
      @ Out, first matching node or None if no node was found
    '''
    if path[:1] == "/":
      path = "." + path
    return self._rootnode.find(path)

  def findall(self, path):
    '''
      Method to find the all toplevel nodes with a given name
      @ In, path, string, the path or name
      @ Out, A list or iterator containing all matching nodes
    '''
    if path[:1] == "/":
      path = "." + path
    return self._rootnode.findall(path)

  def iterfind(self, path):
    '''
      Method to find the all matching subnodes with a given name
      @ In, path, string, the path or name
      @ Out, a sequence of node instances
    '''
    if path[:1] == "/":
      path = "." + path
    return self._rootnode.iterfind(path)

  def writeNodeTree(self,dumpFile):
    '''
      This method is used to write the content of the whole tree into a file
      @ In, file instance or string, filename (string) or file instance(opened file)
    '''
    if type(dumpFile) in [str,unicode,bytes]: myFile = open(dumpFile,'w')
    else                                    : myFile = dumpFile
    myFile.write('<NodeTree name = "'+self._rootnode.name+'">\n')
    self._rootnode.writeNode(myFile)
    myFile.write('</NodeTree>\n')
    if type(dumpFile) in [str,unicode,bytes]: myFile.close()

  def stringNodeTree(self,msg=''):
    msg=str(msg)
    msg=self._rootnode.stringNode(msg)
    return msg

####################
#  NodePath Class  #
#  used to iterate #
####################
class NodePath(object):
  def find(self, node, name):
    for nod in node:
      if nod.name == name:
        return nod
    return None
  def iterfind(self, node, name):
    if name[:3] == ".//":
      for nod in node.iter(name[3:]):
        yield nod
      for nod in node:
        if nod.name == name:
          yield nod
  def findall(self, node, name):
    return list(self.iterfind(node, name))

def isnode(node):
  return isinstance(node, Node) or hasattr(node, "name")
