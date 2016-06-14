'''
Tools used to format, edit, and print XML in a RAVEN-like way
talbpaul, 2016-05
'''

from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom as pxml

def prettify(tree):
  """
    Script for turning XML tree into something mostly RAVEN-preferred.  Does not align attributes as some devs like (yet).
    The output can be written directly to a file, as file('whatever.who','w').writelines(prettify(mytree))
    @ In, tree, xml.etree.ElementTree object, the tree form of an input file
    @Out, towrite, string, the entire contents of the desired file to write, including newlines
  """
  #make the first pass at pretty.  This will insert way too many newlines, because of how we maintain XML format.
  pretty = pxml.parseString(ET.tostring(tree.getroot())).toprettyxml(indent='  ')
  #loop over each "line" and toss empty ones, but for ending main nodes, insert a newline after.
  toWrite=''
  for line in pretty.split('\n'):
    if line.strip()=='':
      continue
    toWrite += line.rstrip()+'\n'
    if line.startswith('  </'):
      toWrite+='\n'
  return toWrite

def newNode(tag,text='',attrib={}):
  """
    Creates a new node with the desired tag, text, and attributes more simply than can be done natively.
    @ In, tag, string, the name of the node
    @ In, text, string, optional, the text of the node
    @ In, attrib, dict{string:string}, attribute:value pairs
    @ Out, el, xml.etree.ElementTree.Element, new node
  """
  el = ET.Element(tag,attrib=attrib)
  el.text = text
  return el

def newTree(name):
  """
    Creates a new tree with named node as its root
    @ In, name, string, name of root node
    @ Out, tree, xml.etree.ElementTree.ElementTree, tree
  """
  tree = ET.ElementTree(element=newNode(name))
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
      return oneUp.find(path[-1])
    else:
      return None
  else:
    return root.find(path[-1])

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
