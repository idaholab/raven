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

from .utils import toString, getRelativeSortedListEntry
import xml.etree.ElementTree as ET
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

def prettify(tree, doc=False, docLevel=0, startingTabs=0, addRavenNewlines=True):
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
  def prettifyNode(node, tabs=0, ravenNewlines=True):
    """
      "prettifies" a single node, and calls the same for its children
      adds whitespace to make node more human-readable
      @ In, node, ET.Element, node to prettify
      @ In, tabs, int, optional, indentation level for this node in the global scheme
      @ In, addRavenNewlines, bool, optional, if True then adds newline space between each main-level entity
      @ Out, None
    """
    linesep = "\n" #os.linesep
    child = None #putting it in namespace
    space = ' '*2*tabs
    newlineAndTab = linesep+space
    if node.text is None:
      node.text = ''
    if len(node):
      node.text = node.text.strip()
      if doc and tabs < docLevel and node.text=='...':
        node.text = newlineAndTab+'  '+node.text+newlineAndTab+'  '
      else:
        node.text = node.text + newlineAndTab+'  '
      for child in node:
        prettifyNode(child, tabs+1, ravenNewlines=ravenNewlines)
      #remove extra tab from last child
      child.tail = child.tail[:-2]
    if node.tail is None:
      node.tail = ''
      if doc and tabs!=0 and tabs < docLevel + 1:
        node.tail = newlineAndTab + '...'
    else:
      node.tail = node.tail.strip()
      if doc and tabs < docLevel + 1:
        node.tail += newlineAndTab + '...'
    #custom: RAVEN likes spaces between first-level tab objects
    if ravenNewlines and tabs == 1 and not isComment(node):
      lines = linesep + linesep
    else:
      lines = linesep
    node.tail = node.tail + lines + space
    #custom: except if you're the last child
    if ravenNewlines and tabs == 0 and child is not None:
      child.tail = child.tail.replace(linesep + linesep, linesep)
  #end prettifyNode
  if isinstance(tree, ET.ElementTree):
    prettifyNode(tree.getroot(), tabs=startingTabs, ravenNewlines=addRavenNewlines)
    # NOTE must use utils.toString because ET.tostring returns bytestring in python3
    #  -- if ever we drop python2 support, can use ET.tostring(xml, encoding='unicode')
    return toString(ET.tostring(tree.getroot()))
  else:
    # NOTE must use utils.toString because ET.tostring returns bytestring in python3
    prettifyNode(tree, tabs=startingTabs, ravenNewlines=addRavenNewlines)
    return toString(ET.tostring(tree))

def newNode(tag, text='', attrib=None):
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
  for key, value in attrib.items():
    value = str(value)
    cleanAttrib[fixXmlText(key)] = fixXmlText(value)
  el = ET.Element(tag, attrib=cleanAttrib)
  el.text = fixXmlText(text)
  return el

def newTree(name, attrib=None):
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
  wildcards = ['*', '.', '//', '..']
  path = _path[:]
  found = path.split('/')
  for i, f in enumerate(found):
    # modifier?
    if '[' in f:
      tag, mod = f.split('[')
      mod = mod.rstrip(' ]')
      #find out what type "mod" is
      if mod.startswith('@'):
        #dealing with attributes
        if '=' in mod:
          # have an attribute and a value
          attrib, val = mod.split('=')
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
        tagg, text = mod.split('=')
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

def findPath(root, path):
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

def findPathEllipsesParents(root, path, docLevel=0):
  """
    As with findPath, but the parent nodes are kept and ellipses text are used to replace siblings in the resulting tree.
    @ In, root, xml.etree.ElementTree.Element, the node to start searching along
    @ In, path, string, |-seperated xml path (as "Simulation|RunInfo|JobName")
    @ In, docLevel, int, optional, if doc then only this many levels of tabs will use ellipses documentation
    @ Out, newRoot, None or xml.etree.ElementTree.Element, None if not found or element if found
  """
  foundNode = findPath(root, path)
  if foundNode is None:
    return None
  newRoot = newNode(root.tag, text='...')
  curNode = newRoot
  path = path.split('|')[:-1]
  for e, entry in enumerate(path):
    text = ''
    if e < docLevel:
      text = '...'
    nNode = newNode(entry, text=text)
    curNode.append(nNode)
    curNode = nNode
  curNode.append(foundNode)
  return newRoot

def loadToTree(filename, preserveComments=False):
  """
    loads a file into an XML tree
    @ In, filename, string, the file to load
    @ In, preserveComments, bool, optional, if True then preserve comments in XML tree
    @ Out, root, xml.etree.ElementTree.Element, root of tree
    @ Out, tree, xml.etree.ElementTree.ElementTree, tree read from file
  """
  if preserveComments:
    parser = ET.XMLParser(target=CommentedTreeBuilder())
  else:
    parser = None
  tree = ET.parse(filename, parser=parser)
  root = tree.getroot()
  return root, tree

def fixXmlText(msg):
  """
    Removes unallowable characters from xml
    @ In, msg, string, tag/text/attribute
    @ Out, msg, string, fixed string
  """
  # replace illegal characters with "?"
  # from http://boodebr.org/main/python/all-about-python-and-unicode#UNI_XML
  RE_XML_ILLEGAL = u'([\u0000-\u0008\u000b-\u000c\u000e-\u001f\ufffe-\uffff])' + \
                 u'|' + \
                 u'([%s-%s][^%s-%s])|([^%s-%s][%s-%s])|([%s-%s]$)|(^[%s-%s])' % \
                  ('\ud800', '\udbff', '\udc00', '\udfff',
                   '\ud800', '\udbff', '\udc00', '\udfff',
                   '\ud800', '\udbff', '\udc00', '\udfff')
  try:
    msg = re.sub(RE_XML_ILLEGAL, "?", msg)
  except TypeError:
    pass # not a string, so don't replace illegals, just pass on.
  return msg

def fixXmlTag(msg):
  """
    Does the same things as fixXmlText, but with additional tag restrictions.
    @ In, msg, string, tag/text/attribute
    @ Out, msg, string, fixed string
  """
  #define some presets
  letters = u'([a-zA-Z])'
  notAllTagChars = '(^[a-zA-Z0-9-_.]+$)'
  notTagChars = '([^a-zA-Z0-9-_.])'
  #rules:
  #  1. Can only contain letters, digits, hyphens, underscores, and periods
  try:
    matched = re.match(notAllTagChars, msg)
  except TypeError:
    return msg # not a string, so don't continue
  if not bool(matched):
    pre = msg
    msg = re.sub(notTagChars, '.', msg)
    print('( XML  UTILS ) Replacing illegal tag characters in "{}": {}'.format(pre, msg))
  #  2. Start with a letter or underscore
  if not bool(re.match(letters + u'|([_])', msg[0])) or bool(re.match(u'([xX][mM][lL])', msg[:3])):
    print('( XML  UTILS ) Prepending "_" to illegal tag "' + msg + '"')
    msg = '_' + msg
  return msg

def expandExternalXML(root, workingDir):
  """
    Expands "ExternalXML" nodes with the associated nodes and returns the full tree.
    @ In, root, xml.etree.ElementTree.Element, main node whose children might be ExternalXML nodes
    @ In, workingDir, string, base location from which to find additional xml files
    @ Out, None
  """
  # find instances of ExteranlXML nodes to replace
  for i, subElement in enumerate(root):
    if subElement.tag == 'ExternalXML':
      nodeName = subElement.attrib['node']
      xmlToLoad = subElement.attrib['xmlToLoad'].strip()
      root[i] = readExternalXML(xmlToLoad, nodeName, workingDir)
    # whether expanded or not, search each subnodes for more external xml
    expandExternalXML(root[i], workingDir)

def readExternalXML(extFile, extNode, cwd):
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
    extFile = os.path.join(cwd, extFile)
  if not os.path.exists(extFile):
    raise IOError('XML UTILS ERROR: External XML file not found: "{}"'.format(os.path.abspath(extFile)))
  # find the element to read
  try:
    root = ET.parse(extFile).getroot()
  except ET.ParseError as err:
    lineNo, col = err.position
    with open(extFile, 'r') as inFile:
      content = inFile.readlines()
    line = content[lineNo-1].strip('\n')
    caret = '{:=>{}}'.format('^', col)
    err.msg = '{}\n{}\n{}\n in input file: {}'.format(err, line, caret, extFile)
    raise err
  if root.tag != extNode.strip():
    raise IOError('XML UTILS ERROR: Node "{}" is not the root node of "{}"!'.format(extNode, extFile))
  return root

def replaceVariableGroups(node, variableGroups):
  """
    Replaces variables groups with variable entries in text of nodes
    @ In, node, xml.etree.ElementTree.Element, the node to search for replacement
    @ In, variableGroups, dict, variable group mapping
    @ Out, None
  """
  if node.text is not None and node.text.strip() != '':
    textEntries = list(t.strip() for t in node.text.split(','))
    for t,text in enumerate(textEntries):
      if text in variableGroups.keys():
        textEntries[t] = variableGroups[text].getVarsString()
        print('( XML  UTILS ) Replaced text in <%s> with variable group "%s"' %(node.tag,text))
    #note: if we don't explicitly convert to string, scikitlearn chokes on unicode type
    node.text = str(','.join(textEntries))
  for child in node:
    replaceVariableGroups(child, variableGroups)

def findAllRecursive(node, element):
  """
    A function for recursively traversing a node in an elementTree to find
    all instances of a tag.
    Note that this method differs from findall() since it goes for all nodes,
    subnodes, subsubnodes etc. recursively
    @ In, node, ET.Element, the current node to search under
    @ In, element, str, the string name of the tags to locate
    @ Out, result, list, a list of the currently recovered results
  """
  result = []
  for elem in node.iter(tag=element):
    result.append(elem)
  return result

def toFile(name, root, pretty=True):
  """
    Writes out XML element "root" to file named "name". By default, applies prettifier.
    @ In, name, str, name of destination file
    @ In, root, xml.etree.ElementTree.Element, node to write
    @ In, pretty, bool, optional, whether to prettify tree
    @ Out, None
  """
  if pretty:
    s = prettify(root)
  with open(os.path.abspath(os.path.expanduser(name)), 'w') as f:
    f.write(s)
#
# XML Reader Customization
#
#
class CommentedTreeBuilder(ET.TreeBuilder):
  """
    Comment-preserving tree reader.
    Taken from https://stackoverflow.com/questions/33573807/faithfully-preserve-comments-in-parsed-xml-python-2-7
  """
  def __init__(self, *args, **kwargs):
    super(CommentedTreeBuilder, self).__init__(*args, **kwargs)
    # self._parser.CommentHandler = self.comment

  def comment(self, data):
    """
      Typifies comments in the XML tree
      @ In, data, instance, internal ElementTree data structure
      @ Out, None
    """
    self.start(ET.Comment, {})
    self.data(data)
    self.end(ET.Comment)

#
# Classes for standardized RAVEN XML writing (outputs of DataObjects, ROMs, etc)
#
#
class StaticXmlElement(object):
  """
    Standardized RAVEN output XML structure for values who do not depend on any index (scalars)
    Example:
    <root type='Static'>
      <parameter>
        <single-value properties> value </single-value properties>
        <multi-value properties>
          <w.r.t. parameter2> value </w.r.t. paramter2>
          <w.r.t. parameter2> value </w.r.t. paramter2>
        </multi-value properties>
      <parameter>
    </root>
  """
  def __init__(self, tag, attrib=None, rootType='Static'):
    """
      Constructor.
      @ In, tag, string, name for root node ('root' in structure example in class docstrings)
      @ In, attrib, dict, optional, attributes for root node
      @ In, rootType, str, optional, type as a string
      @ Out, None
    """
    # default attrib to empty dictionary
    if attrib is None:
      attrib = {}
    # for future reading with RAVEN, mark as a static node
    if 'type' not in attrib:
      attrib['type'] = rootType
    # initialize class variables
    self._tree = newTree(tag, attrib)    # base tree structure
    self._root = self._tree.getroot()   # root element of tree

  def addScalar(self, target, name, value, root=None, attrs=None, replaceNode=False):
    """
      Adds a node entry named "name" with value/text "value" to a node "target". For example:
      <root>
        <target>
          <name>value</name>
        </target>
      </root>
      @ In, target, string, name of existing or new node to be added
      @ In, name, string, name of new subnode to be added to node
      @ In, value, string, text of new subnode
      @ In, root, xml.etree.ElementTree.Element, optional, root to append to
      @ In, attrs, dict, optional, attributes for new subnode
      @ In, replaceNode, bool, optional, replace node if found in the tree already?
      @ Out, None
    """
    if root is None:
      root = self.getRoot()
    # find target node (if it exists, otherwise create it)
    targ = self._findTarget(root, target) if root.tag != target.strip() else root
    if replaceNode:
      el = targ.find(name)
      if el is not None:
        targ.remove(el)
    targ.append(newNode(name, text=value, attrib=attrs))

  def addVector(self, target, name, valueCont, root=None, attrs=None,
                valueAttrsDict=None, replaceNode=False):
    """
      Adds a node entry named "name" with value "value" to "target" node, such as
      <root>
        <target>
          if valueCont is a dict:
             <name>
               <with_respect_to_name1> value 1 </with_respect_to_name1>
               <with_respect_to_name2> value 2 </with_respect_to_name2>
               <with_respect_to_name3> value 3 </with_respect_to_name3>
             </name>
           else:
             <name> value </name>
        </target>
      </root>
      The valueCont should be as {with_respect_to_name1: value1, with_respect_to_name2: value2, etc}
      For example, if the "name" is sensitivity_coefs, each entry would be the sensitivity of the "target"
        to "with_respect_to_name1" and etc.
      @ In, target, string, target parameter to add node value to
      @ In, name, string, name of characteristic of target to add
      @ In, valueCont, dict or str, if dict:
                                        name:value dictionary of metric values
                                     else:
                                        value of node "name"
      @ In, root, xml.etree.ElementTree.Element, optional, node to append to
      @ In, attrs, dict, optional, dictionary of attributes to be stored in the node (name)
      @ In, valueAttrsDict, dict, optional, dictionary of attributes to be stored along the subnodes
            identified by the valueCont dictionary
      @ In, replaceNode, bool, optional, replace node (named "name") if found in the tree already?
      @ Out, None
    """
    isStr = isinstance(valueCont, str)
    if root is None:
      root = self.getRoot()
    if valueAttrsDict is None:
      valueAttrsDict = {}
    targ = self._findTarget(root, target) if root.tag != target.strip() else root
    if replaceNode:
      # replace node?
      el = targ.find(name)
      if el is not None:
        targ.remove(el)
    nameNode = newNode(name, attrib=attrs, text=valueCont if isStr else '')
    if not isStr:
      for key, value in sorted(list(valueCont.items())):
        nameNode.append(newNode(key, text=value, attrib=valueAttrsDict.get(key, None)))
    targ.append(nameNode)

  def getRoot(self):
    """
      Getter for root node.
      @ In, None
      @ Out, xml.etree.ElementTree.Element, root node
    """
    return self._root

  def _findTarget(self, root, target):
    """
      Searches "root" for "target" node and makes it if not found
      @ In, root, xml.etree.ElementTree.Element, node to search under
      @ In, target, string, name of target to find
      @ Out, targ, xml.etree.ElementTree.Element, desired taret node
    """
    # find target node
    targ = findPath(root, target)
    # if it doesn't exist, create it
    if targ is None:
      targ = newNode(target)
      root.append(targ)
    return targ

def staticFromString(s):
  """
    Parse string as XML.
    @ In, s, str, XML in string format
    @ Out, new, StaticXmlElement, xml
  """
  new = StaticXmlElement('temp')
  new._root = ET.fromstring(s)
  new._tree = ET.ElementTree(element=new._root)
  return new

#
# Dynamic version
#
#
class DynamicXmlElement(StaticXmlElement):
  """
    <root type='Static'>
      <pivot value="value">
        <parameter>
          <single-value properties>value</single-value properties>
          <multi-value properties>
            <w.r.t. parameter2>value</w.r.t. paramter2>
            <w.r.t. parameter2>value</w.r.t. paramter2>
          </multi-value properties>
        <parameter>
    </root>
  """
  def __init__(self, tag, attrib=None, rootType='Dynamic', pivotParam=None):
    """
      Constructor.
      @ In, tag, string, name for the root node
      @ In, attrib, dict, optional, attributes for root node
      @ In, rootType, str, optional, type as a string
      @ Out, None
    """
    StaticXmlElement.__init__(self, tag, attrib, rootType)
    if pivotParam is None:
      raise IOError('Initializing xmlUtils.DynamicXmlElement, and no pivotParam was provided!')
    self.pivotParam = pivotParam
    self.pivotNodes = []
    self.pivotVals = []

  def addScalar(self, target, name, value, pivotVal, attrs=None, general=False):
    """
      Adds a node entry named "name" with value "value" to "target" node, such as
      <root>
        <pivotParam value=pivotVal>
          <target>
            <name>value<name>
      @ In, target, string, target parameter to add node value to
      @ In, name, string, name of characteristic of target to add
      @ In, value, string/float/etc, value of characteristic
      @ In, pivotVal, float, value of the pivot parameter
      @ In, attrs, dict, optional, dictionary containing the attributes to be stored in the node
      @ In, general, bool, optional, if True then use the "main" pivotless node, instead of the pivot.
      @ Out, None
    """
    # if writing general (not time-specific data), then we write to a particular node
    if general:
      pivotNode = StaticXmlElement._findTarget(self, self._root, 'general')
    # otherwise, we write to the appropriate time-dependent node
    else:
      pivotNode = self._findPivotNode(pivotVal)
    StaticXmlElement.addScalar(self, target, name, value, root=pivotNode, attrs=attrs)

  def addScalarNode(self, node, pivotVal):
    """
      Places an already-constructed XML node under a pivot value node.
      @ In, node, xml.etree.ElementTree.Element, node to add under pivot node
      @ In, pivotVal, float, value of pivot where node should be placed
      @ Out, None
    """
    pivot = self._findPivotNode(pivotVal)
    # TODO merge existing nodes if present? Future work.
    pivot.append(node)

  def _findPivotNode(self, pivotVal):
    """
      Searches pivot node for node with value pivotVal, or adds it if it doesn't exist
      @ In, pivotVal, float, value of pivot to find
      @ Out, pivotNode, xml.etree.ElementTree.Element, node desired
    """
    _, pivotIndex, pivotVal = getRelativeSortedListEntry(self.pivotVals, pivotVal, tol=1e-10)
    # check if insertion needs to be performed
    if len(self.pivotVals) > len(self.pivotNodes):
      # create new node
      pivotNode = newNode(self.pivotParam, attrib={'value':pivotVal})
      self.pivotNodes.insert(pivotIndex, pivotNode)
      self._root.append(pivotNode)
    else:
      pivotNode = self.pivotNodes[pivotIndex]
    return pivotNode
