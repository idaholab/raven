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
This is a simplified version of Raven framework/utils/TreeStructure.py
"""

#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#message handler
import os, sys
#import MessageHandler

##################
# MODULE METHODS #
##################

def __split_line_comment(line):
  """
    Splits a line into the data part and the comment part
    @ In, line, string, line
    @ Out, (data, comment), (string, string), The comment part maybe empty
  """
  data = ''
  comment = ''
  in_string = False
  in_comment = False
  for char in line:
    if char == "'":
      in_string = not in_string
    if char == '#' and not in_string:
      in_comment = True
    if in_comment:
      comment += char
    else:
      data += char
  return (data, comment)

def getpot_to_input_node(getpot):
  """
    Converts a getpot file into an InputNode object.
    @ In, getpot, file, file object with getpot syntax
    @ Out, tree, InputNode, node with sorted information
  """
  #root = Node()
  parentNodes = []#root]
  currentNode = None
  global comment
  comment = None
  def addComment(node):
    """
      If comment is not None, adds it to node
      @ In, node, Node, node
      @ Out, None
    """
    global comment
    if comment is not None:
      node.addComment(comment)
      comment = None
  #end addComment
  for line in getpot:
    line = line.strip()
    #if comment in line, store it for now
    line, new_comment = __split_line_comment(line)
    if len(new_comment) > 0:
      if comment is not None:
        #need to stash comments, attributes for node
        comment += "\n"+new_comment
      comment = new_comment
    #if starting new node
    if line.startswith('[./') and line.endswith(']'):
      #if child node, stash the parent for now
      if currentNode is not None:
        parentNodes.append(currentNode)
      currentNode = InputNode(tag=line[3:-1])#,attrib={})
      addComment(currentNode)
    #if at end of node
    elif line == '[../]':
      #FIXME what if parentNodes is empty, i.e., back to the beginning?  Simulation node wrapper?
      #add currently-building node to its parent
      if len(parentNodes)>0:
        parentNodes[-1].append(currentNode)
        #make parent the active node
        currentNode = parentNodes.pop()
      else:
        #this is the root
        root = currentNode
      addComment(currentNode) #FIXME should belong to next child? Hard to say.
    #empty line
    elif line == '':
      if currentNode is not None:
        currentNode.addComment(comment)
    #attribute setting line
    elif '=' in line:
      #TODO FIXME add attribute comment!
      equal_index = line.find('=')
      attribute = line[:equal_index].strip()
      value = line[equal_index+1:].strip()
      value = value.strip("'")
      if attribute in currentNode.attrib.keys():
        raise IOError('Multiple attributes defined with same name! "'+attribute+'" = "'+value+'"')
      #special keywords: "name" and "value"
      elif attribute == 'value':
        currentNode.text = value
        #TODO default lists: spaces, commas, or ??? (might just work anyway?)
        # -> getpot uses spaces surrounded by apostrophes, '1 2 3'
        # -> raven sometimes does spaces <a>1 2 3</> and sometimes commas <a>1,2,3</a>
      else:
        currentNode.attrib[attribute] = value
    #[]
    elif line == "[]":
      assert parentNodes == [], line
      root = currentNode
      currentNode = None
    #[something]
    elif line.startswith('[') and line.endswith(']'):
      assert currentNode is None, line
      currentNode = InputNode(tag=line[1:-1])
      addComment(currentNode)
    else:
      addComment(currentNode)
      raise IOError('Unrecognized line syntax:',line)
  return root


#########
# NODES #
#########
class InputNode:
  """
    Node in an input tree.  Simulates all the behavior of an XML node.
  """
  #built-in functions
  def __init__(self,tag='',attrib=None,text='',comment=None):
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
