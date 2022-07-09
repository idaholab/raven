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
Created on Jan 28, 2014
@ author: alfoa
"""

import xml.etree.ElementTree as ET
import os, sys

from . import xmlUtils
#message handler
from ..BaseClasses import MessageUser

##################
# MODULE METHODS #
##################
class InputParsingError(IOError):
  """
    Specific error for input parsing problems
  """
  pass

def dump(node):
  """
    Write a node or tree to stdout.
    @ In, node, InputNode, node to print
    @ Out, dump, string, string representation of node
  """
  print(node.printXML())

def parse(inFile,dType=None):
  """
    Given a file (XML), process it into a node.
    @ In, inFile, file, file to process, or string acceptable
    @ In, dType, string, optional, type of processing to use (xml or getpot)
    @ Out, tree, InputTree, structured input
  """
  #check type of file, and open if it is a string.
  if type(inFile).__name__ in ['str','bytes','unicode']:
    inFile = open(inFile,'r')
  if dType is None:
    extension = inFile.name.split('.')[-1].lower()
    if extension == 'xml':
      dType = 'xml'
    else:
      #Possibly we should just try parsing the file instead of checking?
      raise InputParsingError('Unrecognized file type for:',inFile,' | Expected .xml')
  if dType.lower()=='xml':
    try:
      parser = ET.XMLParser(target=CommentedTreeBuilder())
      xmltree = ET.parse(inFile,parser=parser) #parser is defined below, under XMLCommentParser
      tree = xmlToInputTree(xmltree)
    except ET.ParseError as err:
      fileName = inFile.name
      inFile.close()
      lineNo, col = err.position
      with open(fileName, 'r') as inFile:
        content = inFile.readlines()
      line = content[lineNo-1].strip('\n')
      caret = '{:=>{}}'.format('^', col)
      err.msg = '{}\n{}\n{}\n in input file: {}'.format(err, line, caret, fileName)
      raise err
  else:
    raise InputParsingError('Unrecognized file type for:',inFile)
  return tree

def tostring(node):
  """
    Generates a string representation of the tree, in a format determined by the user.
    @ In, node, InputNode or InputTree, item to turn into a string
    @ Out, tostring, string, full tree in string form
  """
  if isinstance(node,InputNode) or isinstance(node,InputTree):
    return node.printXML()
  else:
    raise NotImplementedError('TreeStructure.tostring received "'+str(node)+'" but was expecting InputNode or InputTree.')

def xmlToInputTree(xml):
  """
    Converts an XML tree into an InputTree object.
    @ In, xml, xml.etree.ElementTree, tree to convert
    @ Out, tree, NodeTree, tree with sorted information
  """
  xmlRoot = xml.getroot() #TODO what if they hand in element instead of tree?
  rootName = xmlRoot.tag
  rootAttrib = xmlRoot.attrib
  rootText = xmlRoot.text.strip()
  tsRoot = InputNode(rootName,rootAttrib,rootText)
  def readChild(parent,child,commentsToAdd):
    """
      reads child from XML node and appends it to parent
      @ In, parent, InputNode, parent whose children are being added
      @ In, child, InputNode, child being read
      @ In, commentsToAdd, list, comment objects that have been saved
      @ Out, commentsToAdd, list, comments objects to preserve
    """
    #case: comment
    if type(child.tag).__name__ == 'function':
      commentsToAdd.append(child)
    #case: normal entry
    else:
      #childNode is the TreeStructure node that will replicate "child"
      #clear out extra space from formatting
      if child.text is not None:
        child.text = child.text.strip()
      #create child node
      childNode = InputNode(child.tag,attrib=child.attrib,text=child.text)
      #handle comments collected
      for comment in commentsToAdd:
        if ':' in comment.text:
          attribName = comment.text.split(':')[0].strip()
          if attribName in childNode.attrib.keys():
            childNode.setAttribComment(':'.join(comment.text.split(':')[1:]))
          else:
            childNode.addComment(comment.text)
        else:
          childNode.addComment(comment.text)
      #clear existing comments
      commentsToAdd = []
      #add children of current child node
      for cchild in child:
        commentsToAdd = readChild(childNode,cchild,commentsToAdd)
      #add childNode to the parent
      parent.append(childNode)
    return commentsToAdd
  #end readChild
  commentsToAdd = []
  for child in xml.__dict__['_root']:
    commentsToAdd = readChild(tsRoot,child,commentsToAdd)
  return InputTree(tsRoot)

def inputTreeToXml(ts,fromNode=False):
  """
    Converts InputTree into XML
    @ In, ts, InputTree, tree to convert
    @ In, fromNode, bool, if True means input is a node instead of a tree
    @ Out, xTree, xml.etree.ElementTree.ElementTree, tree in xml style
  """
  if fromNode:
    tRoot = ts
  else:
    tRoot = ts.getroot()
  xRoot = xmlUtils.newNode(tRoot.tag,tRoot.text,tRoot.attrib)
  def addChildren(xNode,tNode):
    """
      Adds children of tnode to xnode, translating
      @ In, xnode, xml.etree.ElementTree.Element, xml node
      @ In, tnode, Node, tree structure node
      @ Out, None
    """
    for child in tNode:
      #TODO case: comment
      for comment in child.comments:
        if comment is not None and len(comment.strip())>0:
          #append comments before children
          xNode.append(ET.Comment(comment))
      if child.text is None:
        child.text = ''
      childNode = xmlUtils.newNode(child.tag,text=child.text,attrib=child.attrib)
      xNode.append(childNode)
      addChildren(childNode,child)
  #end addChildren
  addChildren(xRoot,tRoot)
  if fromNode:
    return xRoot
  else:
    return ET.ElementTree(element=xRoot)

###########
# PARSERS #
###########
## Extracted from: https://stackoverflow.com/questions/33573807/faithfully-preserve-comments-in-parsed-xml-python-2-7
## As mentioned on this post, we could also potentially use lxml to handle this
## automatically without the need for a special class
class CommentedTreeBuilder(ET.TreeBuilder):
  """
      A class for preserving comments faithfully when parsing XML
  """
  def __init__(self, *args, **kwargs):
    """
        The constructor that passes arguments to the
        xml.etree.ElementTree.TreeBuilder class.
        See the relevant documentation for the arguments accepted by the
        base class's __init__ function.
    """
    super(CommentedTreeBuilder, self).__init__(*args, **kwargs)

  def comment(self, data):
    """
        A function for appropriately surrounding data with the necessary
        markers to establish it as a comment.
        @ In, data, string, the text that needs to be wrapped in a comment.
        @ Out, None
    """
    self.start(ET.Comment, {})
    self.data(data)
    self.end(ET.Comment)

#########
# NODES #
#########
class InputNode:
  """
    Node in an input tree.  Simulates all the behavior of an XML node.
  """
  #built-in functions
  def __init__(self, tag='', attrib=None, text='', comment=None):
    """
      Constructor.
      @ In, tag, string, node name
      @ In, attrib, dict, attributes
      @ In, text, string, text of node
      @ Out, None
    """
    if attrib is None:
      self.attrib = {}
    else:
      self.attrib = attrib #structure: {name: {value='',comment=''}}

    self.tag = tag       #node name, in XML known as "tag"
    self.text = text     #node text, same in XML
    self.children = []   #branches off of this node
    self.comments = [comment] if comment is not None else [] #allow for multiple comments

  def __eq__(self,other):
    """
      Determines if this object is NOT the same as "other".
      @ In, other, the object to compare to
      @ Out, same, boolan, true if same
    """
    if isinstance(other,self.__class__):
      same = True
      if self.tag != other.tag or \
             self.text != other.text or \
             self.attrib != other.attrib:
        same = False
      #else: TODO compare children!
      #TODO use XML differ for this whole thing?
      return same
    return False

  def __ne__(self,other):
    """
      Determines if this object is NOT the same as "other".
      @ In, other, the object to compare to
      @ Out, same, boolan, true if not same
    """
    return not self.__eq__(other)

  def __hash__(self):
    """
      Overrides the default hash.
      @ In, None
      @ Out, hash, tuple, name and values and text
    """
    return hash(tuple((self.tag,tuple(sorted(self.attrib.items())),self.text)))

  def __iter__(self):
    """
      Provides a method to iterate over the child nodes of this node.
      @ In, None
      @ Out, __iter__, iterator, generator for the children
    """
    i = 0
    while i < len(self):
      yield self.children[i]
      i += 1

  def __len__(self):
    """
      Returns the number of child nodes for this node.
      @ In, None
      @ Out, __len__, int, number of children
    """
    return len(self.children)

  def __getitem__(self, index):
    """
      Returns a specific child node.
      @ In, index, int, the index for the child
      @ Out, __getitem__, InputNode, the child
    """
    return self.children[index]

  def __setitem__(self,index,value):
    """
      Sets a specific child node.
      @ In, index, int, the index for the child
      @ In, value, Node, the child itself
      @ Out, None
    """
    value = self.assureIsNode(value)
    self.children[index] = value

  def __repr__(self):
    """
      String representation.
      @ In, None
      @ Out, __repr__, string, representation of the object
    """
    return "<Node %s attrib=%s at 0x%x containing %s branches>" %(repr(self.tag),str(self.attrib),id(self),repr(len(self)))

  #methods
  def add(self, key, value):
    """
      Method to add a new value into this node
      If the key is already present, the corresponding value gets updated
      @ In, key, string, id name of this value
      @ In, value, whatever type, the newer value
    """
    self.attrib[key] = value

  def addComment(self,comment):
    """
      Adds comment to node.
      @ In, comment, string, comment to add
      @ Out, None
    """
    if comment is not None:
      self.comments.append(comment)

  def append(self,node):
    """
      Add a new child node to this node.
      @ In, node, Node, node to append to children
      @ Out, None
    """
    node = self.assureIsNode(node)
    self.children.append(node)

  def find(self,nodeName):
    """
      Searches children for node with name matching nodeName.
      @ In, nodeName, string, name to match
      @ Out, node, InputNode, match is present else None
    """
    for child in self:
      if child.tag == nodeName:
        return child
    return None

  def findall(self,nodeName):
    """
      Searches children for node with name matching nodeName.
      @ In, nodeName, string, name to match
      @ Out, nodes, [InputNode], all the matches
    """
    nodes = []
    for child in self:
      if child.tag == nodeName:
        nodes.append(child)
    return nodes

  def get(self,attr):
    """
      Obtains attribute value for "attr"
      @ In, attr, string, name of attribute to obtain
      @ Out, get, string, value if present else None
    """
    return self.attrib.get(attr,None)

  def getiterator(self,tag=None):
    """
      Deprecated.  Used for compatability with xml.etree.ElementTree.getiterator.
      @ In, tag, string, only return tags matching this type in the iteration
      @ Out, iterator, iterator, tree iterator with matching tag
    """

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
    for e in self.children:
      for e in e.iter(name):
        yield e

  def assureIsNode(self,node):
    """
      Takes care of translating XML to Node on demand.
      @ In, node, Node or ET.Element, node to fix up
      @ Out, node, fixed node
    """
    if not isinstance(node,InputNode):
      # if XML, convert to InputNode
      if isinstance(node,ET.Element):
        tree = ET.ElementTree(node)
        node = xmlToInputTree(tree).getroot()
      else:
        raise TypeError('TREE-STRUCTURE ERROR: When trying to use node "{}", unrecognized type "{}"!'.format(node,type(node)))
    return node

  def printXML(self):
    """
      Returns string representation of tree (in XML format).
      @ In, None
      @ Out, msg, string, string representation
    """
    xml = inputTreeToXml(self,fromNode=True)
    return xmlUtils.prettify(xml)

  def remove(self,node):
    """
      Removes a child from the tree.
      @ In, node, InputNode, node to remove
      @ Out, None
    """
    self.children.remove(node)

  def items(self):
    """
      Gets all the children in the tree.
      @ In, None
      @ Out, children, list, list of all the children.
    """
    return self.children

class HierarchicalNode(MessageUser):
  """
    The Node class. It represents the base for each TreeStructure construction
    These Nodes are particularly for heirarchal structures.
  """
  #TODO the common elements between this and InputNode should be abstracted to a Node class.
  def __init__(self, name, valuesIn={}, text='', **kwargs):
    """
      Initialize Tree,
      @ In, name, string, is the node name
      @ In, valuesIn, dict, optional, is a dictionary of values
      @ In, text, string, optional, the node's text, as <name>text</name>
    """
    super().__init__(**kwargs)
    #check message handler is first object
    values         = valuesIn.copy()
    self.name      = name
    self.type      = 'Node'
    self.printTag  = 'Node:<'+self.name+'>'
    self.values    = values
    self.text      = text
    self._branches = []
    self.parentname= None
    self.parent    = None
    self.depth     = 0
    self.iterCounter = 0

  def __eq__(self,other):
    """
      Overrides the default equality check
      @ In, other, object, comparison object
      @ Out, eq, bool, True if both are the same
    """
    if isinstance(other,self.__class__):
      same = True
      if self.name != other.name:
        same = False
      elif self.text != other.text:
        same = False
      elif self.values != other.values:
        same = False
      # TODO ... check parent and children?
      return same
    return NotImplemented

  def __ne__(self,other):
    """
      Overrides the default equality check
      @ In, other, object, comparison object
      @ Out, ne, bool, True if both aren't the same
    """
    if isinstance(other,self.__class__):
      return not self.__eq__(other)
    return NotImplemented

  def __hash__(self):
    """
      Overrides the default hash.
      @ In, None
      @ Out, hash, tuple, name and values and text
    """
    return hash(tuple(self.name,tuple(sorted(self.values.items())),self.text))

  def __iter__(self):
    """
      basic iteration method
      @ In, None
      @ Out, self, Node instance, iterate over self
    """
    i=0
    while i<len(self._branches):
      yield self._branches[i]
      i+=1
    #return self

  def __repr__(self):
    """
      Overload the representation of this object... We want to show the name and the number of branches!!!!
      @ In, None
      @ Out, __repr__, string, the representation of this object
    """
    return "<Node %s values=%s at 0x%x containing %s branches>" % (repr(self.name), str(self.values), id(self), repr(len(self._branches)))

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
      if branchName.strip() == branchv.name:
        isHere = True
    return isHere

  def numberBranches(self):
    """
      Method to get the number of branches
      @ In, None
      @ Out, len, int, number of branches
    """
    return len(self._branches)

  def appendBranch(self, node, updateDepthLocal = False):
    """
      Method used to append a new branch to this node
      @ In, node, Node, the newer node
      @ In, updateDepthLocal, if the depth needs to be updated locally only
      @ Out, None
    """
    node.parentname = self.name
    node.parent     = self
    # this is to avoid max number of recursion if a close loop. TODO: BETTER WAY
    if not updateDepthLocal:
      node.updateDepth()
    else:
      node.depth      = self.depth + 1
    self._branches.append(node)

  def updateDepth(self):
    """
      updates the 'depth' parameter throughout the tree
      @In, None
      @ Out, None
    """
    if self.parent=='root':
      self.depth=0
    else:
      self.depth = self.parent.depth+1
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
    if ego.parentname == 'root':
      result.insert (0, ego)
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
    if len(self.values.keys()) >0:
      dumpFileObj.write(' '+'  '*self.depth +'  <attributes>\n')
    for key,value in self.values.items():
      dumpFileObj.write(' '+'  '*self.depth+'    <'+ key +'>' + str(value) + '</'+key+'>\n')
    if len(self.values.keys()) >0:
      dumpFileObj.write(' '+'  '*self.depth +'  </attributes>\n')
    for e in self._branches:
      e.writeNode(dumpFileObj)
    #if self.numberBranches()>0: # it's unclear why this conditional was here.  It results in the "branch" tag not closing itself. - talbpaul, 2017-10
    dumpFileObj.write(' '+'  '*self.depth + '</branch>\n')

  def stringNode(self,msg=''):
    """
      As writeNode, but returns a string representation of the tree instead of writing it to file.
      @ In, msg, string, optional, the string to populate
      @ Out, msg, string, the modified string
    """
    msg+=''+'  '*self.depth + '<' + self.name + '>'+str(self.text)
    if self.numberBranches()==0:
      msg+='</'+self.name+'>'
    msg+='\n'
    if len(self.values.keys()) >0:
      msg+=''+'  '*self.depth +'  <attributes>\n'
    for key,value in self.values.items():
      msg+=' '+'  '*self.depth+'    <'+ key +'>' + str(value) + '</'+key+'>\n'
    if len(self.values.keys()) >0:
      msg+=''+'  '*self.depth +'  </attributes>\n'
    for e in self._branches:
      msg=e.stringNode(msg)
    if self.numberBranches()>0:
      msg+=''+'  '*self.depth + '</'+self.name+'>\n'
    return msg

##################
#   NODE TREES   #
##################
class InputTree:
  """
    The class that realizes an Input Tree Structure
  """
  #built-in functions
  def __init__(self,rootNode=None):
    """
      Constructor.
      @ In, rootNode, InputNode, optional, root node for tree
      @ Out, None
    """
    self.rootNode = rootNode

  def __repr__(self):
    """
      String representation.
      @ In, None
      @ Out, __repr__, string, representation
    """
    return "{TreeStructure with root node "+str(self.rootNode)+"}"

  #methods
  def getroot(self):
    """
      Method to get root node
      @ In, None
      @ Out, rootNode, Node, root of tree
    """
    return self.rootNode

  def printXML(self):
    """
      Returns string of full XML tree by returning string of root node.
      @ In, None
      @ Out, msg, string, full XML tree
    """
    return self.rootNode.printXML()



class HierarchicalTree(MessageUser):
  """
    The class that realizes a hierarchal Tree Structure
  """
  #TODO the common elements between HierarchicalTree and InputTree should be extracted to a Tree class.
  def __init__(self, node=None):
    """
      Constructor
      @ In, node, Node, optional, the rootnode
      @ Out, None
    """
    super().__init__()
    if not hasattr(self,"type"):
      self.type = 'NodeTree'
    self.printTag  = self.type+'<'+str(node)+'>'
    self._rootnode = node
    if node:
      node.parentname='root'

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
    if path == "root":
      node = self.getrootnode()
    else:
      node = self.find(path)
    if node != None:
      node.name = newName

  def iter(self, name=None):
    """
      Method for creating a tree iterator for the root node
      @ In, name, string, the path or the node name
      @ Out, iter, iterator, the iterator
    """
    if name == 'root':
      return self.__rootnode
    else:
      return self._rootnode.iter(name)

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
      @ Out, find, Node, first matching node or None if no node was found
    """
    if self._rootnode.name == path:
      return self.getrootnode()
    if path[:1] == "/":
      path = "." + path
    return self._rootnode.findBranch(path)

  def findall(self, path):
    """
      Method to find the all toplevel nodes with a given name
      @ In, path, string, the path or name
      @ Out, findall, list of Node iterators, A list or iterator containing all matching nodes
    """
    if self._rootnode.name == path:
      return [self.getrootnode()]
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
    if type(dumpFile).__name__ == 'FileObject':
      myFile = open(dumpFile,'w')
    else:
      myFile = dumpFile
    myFile.write('<NodeTree name = "'+self._rootnode.name+'">\n')
    self._rootnode.writeNode(myFile)
    myFile.write('</NodeTree>\n')
    if type(dumpFile).__name__ == 'FileObject':
      myFile.close()

  def stringNodeTree(self,msg=''):
    """
      As writeNodeTree, but creates a string representation instead of writing to a file.
      @ In, msg, string, the string to populate
      @ Out, msg, string, the populated string
    """
    msg=str(msg)
    msg=self._rootnode.stringNode(msg)
    return msg

##################
# METADATA TREE #
#################
class MetadataTree(HierarchicalTree):
  """
    Class for construction of metadata xml trees used in data objects.  Usually contains summary data
    such as that produced by postprocessor models.  Two types of tree exist: dynamic and static.  See
    RAVEN Output type of Files object.
  """
  #TODO change to inherit from InputTree or base Tree
  def __init__(self, rootName):
    """
      Construct.
      @ In, rootName, str, name of root
      @ Out, None
    """
    self.pivotParam = None
    node = HierarchicalNode(rootName, valuesIn={'dynamic':str(self.dynamic)})
    HierarchicalTree.__init__(self, node)

  def __repr__(self):
    """
      Overridden print method
      @ In, None
      @ Out, repr, string, string of tree
    """
    return self.stringNodeTree()

  def addScalar(self,target,name,value,root=None,pivotVal=None):
    """
      Adds a node entry named "name" with value "value" to "target" node
      Note that Static uses this method exactly, and Dynamic extends it a little
      @ In, target, string, target parameter to add node value to
      @ In, name, string, name of characteristic of target to add
      @ In, value, string/float/etc, value of characteristic
      @ In, root, Node object, optional, node to which "target" belongs or should be added to
      @ In, pivotVal, float, optional, if specified the value of the pivotParam to add target value to
      @ Out, None
    """
    if root is None:
      root = self.getrootnode()
    #FIXME it's possible the user could provide illegal characters here.  What are illegal characters for us?
    targ = self._findTarget(root,target,pivotVal)
    targ.appendBranch(HierarchicalNode(name, text=value))

  def _findTarget(self,root,target,pivotVal=None):
    """
      Used to find target node.  This implementation is specific to static, extend it for dynamic.
      @ In, root, Node object, node to search for target
      @ In, target, string, name of target to find/create
      @ In, pivotVal, float, optional, not used in this method but kept for consistency
      @ Out, tNode, Node object, target node (either created or existing)
    """
    tNode = root.findBranch(target)
    if tNode is None:
      tNode = HierarchicalNode(target)
      root.appendBranch(tNode)
    return tNode



class StaticMetadataTree(MetadataTree):
  """
    Class for construction of metadata xml trees used in data objects.  Usually contains summary data
    such as that produced by postprocessor models.  Two types of tree exist: dynamic and static.  See
    RAVEN Output type of Files object.
  """
  def __init__(self, rootName):
    """
      Constructor.
      @ In, node, Node object, optional, root of tree if provided
      @ Out, None
    """
    self.dynamic = False
    self.type = 'StaticMetadataTree'
    MetadataTree.__init__(self, rootName)




class DynamicMetadataTree(MetadataTree):
  """
    Class for construction of metadata xml trees used in data objects.  Usually contains summary data
    such as that produced by postprocessor models.  Two types of tree exist: dynamic and static.  See
    RAVEN Output type of Files object.
  """
  def __init__(self, rootName, pivotParam):
    """
      Constructor.
      @ In, rootName, str, root of tree if provided
      @ In, pivotParam, str, pivot variable
      @ Out, None
    """
    self.dynamic = True
    self.type = 'DynamicMetadataTree'
    MetadataTree.__init__(self, rootName)
    self.pivotParam = pivotParam

  def _findTarget(self,root,target,pivotVal):
    """
      Used to find target node.  Extension of base class method for Dynamic mode
      @ In, root, Node object, node to search for target
      @ In, target, string, name of target to find/create
      @ In, pivotVal, float, value of pivotParam to use for searching
      @ Out, tNode, Node object, target node (either created or existing)
    """
    pivotVal = float(pivotVal)
    pNode = self._findPivot(root,pivotVal)
    tNode = MetadataTree._findTarget(self,pNode,target)
    return tNode

  def _findPivot(self, root, pivotVal, tol=1e-10):
    """
      Finds the node with the desired pivotValue to the given tolerance
      @ In, root, Node instance, the node to search under
      @ In, pivotVal, float, match to search for
      @ In, tol, float, tolerance for match
      @ Out, pNode, Node instance, matching node
    """
    found = False
    for child in root:
      #make sure we're looking at a pivot node
      if child.name != self.pivotParam:
        continue
      # careful with inequality signs to check for match
      if pivotVal > 0:
        foundCondition = abs(float(child.get('value')) - pivotVal) <= 1e-10*pivotVal
      else:
        foundCondition = abs(float(child.get('value')) - pivotVal) >= 1e-10*pivotVal
      if foundCondition:
        pivotNode = child
        found = True
        break
    #if not found, make it!
    if not found:
      pivotNode = HierarchicalNode(self.pivotParam, valuesIn={'value':pivotVal})
      root.appendBranch(pivotNode)
    return pivotNode


####################
#  NodePath Class  #
#  used to iterate #
####################
class NodePath(object):
  """
    NodePath class. It is used to perform iterations over the HierarchicalTree
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
    @ Out, isnode, bool, is a node instance?
  """
  return isinstance(node, Node) or hasattr(node, "name")
