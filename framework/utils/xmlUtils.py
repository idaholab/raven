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
from .utils import isString
import VariableGroups

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

def newNode(tag,text='',attrib=None):
  """
    Creates a new node with the desired tag, text, and attributes more simply than can be done natively.
    @ In, tag, string, the name of the node
    @ In, text, string, optional, the text of the node
    @ In, attrib, dict{string:string}, attribute:value pairs
    @ Out, el, xml.etree.ElementTree.Element, new node
  """
  if attrib is None:
    attrib = {}
  tag = fixXmlTag(tag)
  text = str(text)
  cleanAttrib = {}
  for key,value in attrib.items():
    value = str(value)
    cleanAttrib[fixXmlText(key)] = fixXmlText(value)
  el = ET.Element(tag,attrib=cleanAttrib)
  el.text = fixXmlText(text)
  return el

def newTree(name,attrib=None):
  """
    Creates a new tree with named node as its root
    @ In, name, string, name of root node
    @ In, attrib, dict, optional, attributes for root node
    @ Out, tree, xml.etree.ElementTree.ElementTree, tree
  """
  if attrib is None:
    attrib = {}
  name = fixXmlTag(name)
  tree = ET.ElementTree(element=newNode(name))
  tree.getroot().attrib = dict(attrib)
  return tree

def fixTagsInXpath(_path):
  """
    Fixes tags/attributes/text in an xml.etree.ElementTree compatible xpath string to use allowable characters
    @ In, _path, str, xpath string
    @ Out, out, str, modified string
  """
  # XPATH OPTIONS:
  # tag wildcards: * . // ..
  # modifiers
  #  [@attrib]
  #  [@attrib='value']
  #  [tag]
  #  [tag='text']
  #  [position]  --> same as [tag] for this purpose
  wildcards = ['*','.','//','..']
  path = _path[:]
  found = path.split('/')
  toRemove = []
  for i,f in enumerate(found):
    # modifier?
    if '[' in f:
      tag,mod = f.split('[')
      mod = mod.rstrip(' ]')
      #find out what type "mod" is
      if mod.startswith('@'):
        #dealing with attributes
        if '=' in mod:
          # have an attribute and a value
          attrib,val = mod.split('=')
          attrib = fixXmlText(attrib[1:])
          val = fixXmlText(val.strip('\'"'))
          mod = '[@'+attrib+"='"+val+"']"
        else:
          # just an attribute
          attrib = fixXmlText(mod.strip('[@] '))
          mod = '[@'+attrib+']'
      # either tag, tag+text, or position
      elif '=' in mod:
        # tag and text
        tagg,text = mod.split('=')
        if tagg not in wildcards:
          tagg = fixXmlTag(tagg.strip())
        text = fixXmlText(text.strip('\'" '))
        mod = '['+tagg+'=\''+text+'\']'
      else:
        # position or tag
        try:
          # position
          int(mod)
        except ValueError:
          # tag
          mod = fixXmlTag(mod)
        mod = '['+mod+']'
    # no mod, just a tag
    else:
      tag = f
      mod = ''
    # tag could be wildcard
    if tag not in wildcards:
      tag = fixXmlTag(tag.strip())
    found[i] = tag+mod
  #reconstruct path
  out = '/'.join(found)
  return out

def findPath(root,path):
  """
    Navigates path to find a particular element
    @ In, root, xml.etree.ElementTree.Element, the node to start searching along
    @ In, path, string, xpath syntax (see for example https://docs.python.org/2/library/xml.etree.elementtree.html#example)
    @ Out, findPath, None or xml.etree.ElementTree.Element, None if not found or first matching element if found
  """
  assert('|' not in path), 'Update XML search to use XPATH syntax!'
  # edit tags for allowable characters
  path = fixTagsInXpath(path)
  found = root.findall(path)
  if len(found) < 1:
    return None
  else:
    return found[0]

def findPathEllipsesParents(root,path,docLevel=0):
  """
    As with findPath, but the parent nodes are kept and ellipses text are used to replace siblings in the resulting tree.
    @ In, root, xml.etree.ElementTree.Element, the node to start searching along
    @ In, path, string, |-seperated xml path (as "Simulation|RunInfo|JobName")
    @ In, docLevel, int, optional, if doc then only this many levels of tabs will use ellipses documentation
    @ Out, newRoot, None or xml.etree.ElementTree.Element, None if not found or element if found
  """
  foundNode = findPath(root,path)
  if foundNode is None:
    return None
  newRoot = newNode(root.tag,text='...')
  curNode = newRoot
  path = path.split('|')[:-1]
  for e,entry in enumerate(path):
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
  if not isString(msg):
    return msg
  #otherwise, replace illegal characters with "?"
  # from http://boodebr.org/main/python/all-about-python-and-unicode#UNI_XML
  RE_XML_ILLEGAL = u'([\u0000-\u0008\u000b-\u000c\u000e-\u001f\ufffe-\uffff])' + \
                 u'|' + \
                 u'([%s-%s][^%s-%s])|([^%s-%s][%s-%s])|([%s-%s]$)|(^[%s-%s])' % \
                  ('\ud800','\udbff','\udc00','\udfff',
                   '\ud800','\udbff','\udc00','\udfff',
                   '\ud800','\udbff','\udc00','\udfff')
  msg = re.sub(RE_XML_ILLEGAL, "?", msg)
  return msg

def fixXmlTag(msg):
  """
    Does the same things as fixXmlText, but with additional tag restrictions.
    @ In, msg, string, tag/text/attribute
    @ Out, msg, string, fixed string
  """
  #if not a string, pass it back through
  if not isString(msg):
    return msg
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
  if not bool(re.match(letters+u'|([_])',msg[0])) or bool(re.match(u'([xX][mM][lL])',msg[:
    3])):
    print('XML UTILS: Prepending "_" to illegal tag "'+msg+'"')
    msg = '_' + msg
  return msg

def expandExternalXML(root,workingDir):
  """
    Expands "ExternalXML" nodes with the associated nodes and returns the full tree.
    @ In, root, xml.etree.ElementTree.Element, main node whose children might be ExternalXML nodes
    @ In, workingDir, string, base location from which to find additional xml files
    @ Out, None
  """
  # find instances of ExteranlXML nodes to replace
  for i,subElement in enumerate(root):
    if subElement.tag == 'ExternalXML':
      nodeName = subElement.attrib['node']
      xmlToLoad = subElement.attrib['xmlToLoad'].strip()
      root[i] = readExternalXML(xmlToLoad,nodeName,workingDir)
    # whether expanded or not, search each subnodes for more external xml
    expandExternalXML(root[i],workingDir)

def readExternalXML(extFile,extNode,cwd):
  """
    Loads external XML into nodes.
    @ In, extFile, string, filename for the external xml file
    @ In, extNode, string, tag of node to load
    @ In, cwd, string, current working directory (for relative paths)
    @ Out, externalElement, xml.etree.ElementTree.Element, object from file
  """
  # expand user tilde
  if '~' in extFile:
    extFile = os.path.expanduser(extFile)
  # check if absolute or relative found
  if not os.path.isabs(extFile):
    extFile = os.path.join(cwd,extFile)
  if not os.path.exists(extFile):
    raise IOError('XML UTILS ERROR: External XML file not found: "{}"'.format(os.path.abspath(extFile)))
  # find the element to read
  root = ET.parse(open(extFile,'r')).getroot()
  if root.tag != extNode.strip():
    raise IOError('XML UTILS ERROR: Node "{}" is not the root node of "{}"!'.format(extNode,extFile))
  return root

def readVariableGroups(xmlNode,messageHandler,caller):
  """
    Reads the XML for the variable groups and initializes them
    @ In, xmlNode, ElementTree.Element, xml node to read in
    @ In, messageHandler, MessageHandler.MessageHandler instance, message handler to assign to the variable group objects
    @ In, caller, MessageHandler.MessageUser instance, entity calling this method (needs to inherit from MessageHandler.MessageUser)
    @ Out, varGroups, dict, dictionary of variable groups (names to the variable lists to replace the names)
  """
  varGroups = {}
  for child in xmlNode:
    varGroup = VariableGroups.VariableGroup()
    varGroup.readXML(child,messageHandler)
    varGroups[varGroup.name]=varGroup
  # initialize variable groups
  while any(not vg.initialized for vg in varGroups.values()):
    numInit = 0 #new vargroups initialized this pass
    for vg in varGroups.values():
      if vg.initialized:
        continue
      try:
        deps = list(varGroups[dp] for dp in vg.getDependencies())
      except KeyError as e:
        caller.raiseAnError(IOError,'Dependency %s listed but not found in varGroups!' %e)
      if all(varGroups[dp].initialized for dp in vg.getDependencies()):
        vg.initialize(varGroups.values())
        numInit+=1
    if numInit == 0:
      caller.raiseAWarning('variable group status:')
      for name,vg in varGroups.items():
        caller.raiseAWarning('   ',name,':',vg.initialized)
      caller.raiseAnError(RuntimeError,'There was an infinite loop building variable groups!')
  return varGroups
