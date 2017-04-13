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
Tools used to format, edit, and print XML in a RAVEN-like way
talbpaul, 2016-05
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom as pxml
import re
import os

#define type checking
def isComment(node):
  """
    Determines if a node is a comment type (by checking its tag).
    @ In, node, xml.etree.ElementTree.Element, node to check
    @ Out, isComment, bool, True if comment type
  """
  if type(node.tag).__name__ == 'function':
    return True
  return False

def prettify(tree,doc=False,docLevel=0,startingTabs=0,addRavenNewlines=True):
  """
    Script for turning XML tree into something mostly RAVEN-preferred.  Does not align attributes as some devs like (yet).
    The output can be written directly to a file, as file('whatever.who','w').writelines(prettify(mytree))
    @ In, tree, xml.etree.ElementTree object, the tree form of an input file
    @ In, doc, bool, optional, if True treats the XML as being prepared for documentation instead of full printing
    @ In, docLevel, int, optional, if doc then only this many levels of tabs will use ellipses documentation
    @ In, startingTabs, int, optional, if provided determines the starting tab level for the prettified xml
    @ In, addRavenNewlines, bool, optional, if True then adds newline space between each main-level entity
    @Out, towrite, string, the entire contents of the desired file to write, including newlines
  """
  def prettifyNode(node,tabs=0,ravenNewlines=True):
    """
      "prettifies" a single node, and calls the same for its children
      adds whitespace to make node more human-readable
      @ In, node, ET.Element, node to prettify
      @ In, tabs, int, optional, indentation level for this node in the global scheme
      @ In, addRavenNewlines, bool, optional, if True then adds newline space between each main-level entity
      @ Out, None
    """
    linesep = os.linesep
    child = None #putting it in namespace
    space = ' '*2*tabs
    newlineAndTab = linesep+space
    if node.text is None:
      node.text = ''
    if len(node)>0:
      node.text = node.text.strip()
      if doc and tabs<docLevel and node.text=='...':
        node.text = newlineAndTab+'  '+node.text+newlineAndTab+'  '
      else:
        node.text = node.text + newlineAndTab+'  '
      for child in node:
        prettifyNode(child,tabs+1,ravenNewlines=ravenNewlines)
      #remove extra tab from last child
      child.tail = child.tail[:-2]
    if node.tail is None:
      node.tail = ''
      if doc and tabs!=0 and tabs<docLevel+1:
        node.tail = newlineAndTab + '...'
    else:
      node.tail = node.tail.strip()
      if doc and tabs<docLevel+1:
        node.tail += newlineAndTab + '...'
    #custom: RAVEN likes spaces between first-level tab objects
    if ravenNewlines and tabs == 1 and not isComment(node):
      lines = linesep + linesep
    else:
      lines = linesep
    node.tail = node.tail + lines + space
    #custom: except if you're the last child
    if ravenNewlines and tabs == 0 and child is not None:
      child.tail = child.tail.replace(linesep+linesep,linesep)
  #end prettifyNode
  if isinstance(tree,ET.ElementTree):
    prettifyNode(tree.getroot(),tabs=startingTabs,ravenNewlines=addRavenNewlines)
    return ET.tostring(tree.getroot())
  else:
    prettifyNode(tree,tabs=startingTabs,ravenNewlines=addRavenNewlines)
    return ET.tostring(tree)


  #### OLD WAY ####
  #make the first pass at pretty.  This will insert way too many newlines, because of how we maintain XML format.
  #pretty = pxml.parseString(ET.tostring(tree.getroot())).toprettyxml(indent='  ')
  #loop over each "line" and toss empty ones, but for ending main nodes, insert a newline after.
  #toWrite=''
  #for line in pretty.split('\n'):
  #  if line.strip()=='':
  #    continue
  #  toWrite += line.rstrip()+'\n'
  #  if line.startswith('  </'):
  #    toWrite+='\n'
  return toWrite

def newNode(tag,text='',attrib={}):
  """
    Creates a new node with the desired tag, text, and attributes more simply than can be done natively.
    @ In, tag, string, the name of the node
    @ In, text, string, optional, the text of the node
    @ In, attrib, dict{string:string}, attribute:value pairs
    @ Out, el, xml.etree.ElementTree.Element, new node
  """
  tag = fixXmlTag(tag)
  text = str(text)
  cleanAttrib = {}
  for key,value in attrib.items():
    value = str(value)
    cleanAttrib[fixXmlText(key)] = fixXmlText(value)
  el = ET.Element(tag,attrib=cleanAttrib)
  el.text = fixXmlText(text)
  return el

def newTree(name,attrib={}):
  """
    Creates a new tree with named node as its root
    @ In, name, string, name of root node
    @ In, attrib, dict, optional, attributes for root node
    @ Out, tree, xml.etree.ElementTree.ElementTree, tree
  """
  name = fixXmlTag(name)
  tree = ET.ElementTree(element=newNode(name))
  tree.getroot().attrib = dict(attrib)
  return tree

def findPath(root,path):
  """
    Navigates path to find a particular element
    @ In, root, xml.etree.ElementTree.Element, the node to start searching along
    @ In, path, string, |-seperated xml path (as "Simulation|RunInfo|JobName")
    @ Out, findPath, None or xml.etree.ElementTree.Element, None if not found or element if found
  """
  path = path.split("|")
  if len(path)>1:
    oneUp = findPath(root,'|'.join(path[:-1]))
    if oneUp is not None:
      toSearch = fixXmlTag(path[-1])
      return oneUp.find(toSearch)
    else:
      return None
  else:
    toSearch = fixXmlTag(path[-1])
    return root.find(toSearch)

def findPathEllipsesParents(root,path,docLevel=0):
  """
    As with findPath, but the parent nodes are kept and ellipses text are used to replace siblings in the resulting tree.
    @ In, root, xml.etree.ElementTree.Element, the node to start searching along
    @ In, path, string, |-seperated xml path (as "Simulation|RunInfo|JobName")
    @ In, docLevel, int, optional, if doc then only this many levels of tabs will use ellipses documentation
    @ Out, newRoot, None or xml.etree.ElementTree.Element, None if not found or element if found
  """
  foundNode = findPath(root,path)
  if foundNode is None: return None
  newRoot = newNode(root.tag,text='...')
  curNode = newRoot
  path = path.split('|')[:-1]
  for e,entry in enumerate(path):
    print('e,entry:',e,entry)
    text = ''
    if e < docLevel:
      text = '...'
    nNode = newNode(entry,text=text)
    curNode.append(nNode)
    curNode = nNode
  curNode.append(foundNode)
  return newRoot



def loadToTree(filename):
  """
    loads a file into an XML tree
    @ In, filename, string, the file to load
    @ Out, root, xml.etree.ElementTree.Element, root of tree
    @ Out, tree, xml.etree.ElementTree.ElementTree, tree read from file
  """
  tree = ET.parse(filename)
  root = tree.getroot()
  return root,tree

def fixXmlText(msg):
  """
    Removes unallowable characters from xml
    @ In, msg, string, tag/text/attribute
    @ Out, msg, string, fixed string
  """
  #if not a string, pass it back through
  if not isinstance(msg,basestring): return msg
  #otherwise, replace illegal characters with "?"
  # from http://boodebr.org/main/python/all-about-python-and-unicode#UNI_XML
  RE_XML_ILLEGAL = u'([\u0000-\u0008\u000b-\u000c\u000e-\u001f\ufffe-\uffff])' + \
                 u'|' + \
                 u'([%s-%s][^%s-%s])|([^%s-%s][%s-%s])|([%s-%s]$)|(^[%s-%s])' % \
                  (unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
                   unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
                   unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff))
  msg = re.sub(RE_XML_ILLEGAL, "?", msg)
  return msg

def fixXmlTag(msg):
  """
    Does the same things as fixXmlText, but with additional tag restrictions.
    @ In, msg, string, tag/text/attribute
    @ Out, msg, string, fixed string
  """
  #if not a string, pass it back through
  if not isinstance(msg,basestring): return msg
  #define some presets
  letters = u'([a-zA-Z])'
  notAllTagChars = '(^[a-zA-Z0-9-_.]+$)'
  notTagChars = '([^a-zA-Z0-9-_.])'
  #rules:
  #  1. Can only contain letters, digits, hyphens, underscores, and periods
  if not bool(re.match(notAllTagChars,msg)):
    pre = msg
    msg = re.sub(notTagChars,'.',msg)
    print('XML UTILS: Replacing illegal tag characters in "'+pre+'":',msg)
  #  2. Start with a letter or underscore
  if not bool(re.match(letters+u'|([_])',msg[0])) or bool(re.match(u'([xX][mM][lL])',msg[:3])):
    print('XML UTILS: Prepending "_" to illegal tag "'+msg+'"')
    msg = '_' + msg
  return msg
