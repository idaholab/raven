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
Getpot/hit input parsing methods
@author: talbpaul
"""
import os
import sys

from ravenframework.utils import utils, TreeStructure

multiIndicators = ('\'', '\"')
validIndicators = ('=', '[')

def preprocessGetpot(getpot):
  """
    Minimize user readability to accentuate computer readability
    Note all comments will be removed, as they aren't computer read
    @ In, getpot, string, string to read in (with newlines, could be open file object)
    @ Out, tree, TreeStructure.NodeTree, tree with information
  """
  # preprocess file
  lines = []
  for line in getpot:
    # drop comments, whitespace
    line = line.split('#', maxsplit=1)[0].strip()
    # "important" standalone entries either are "=" statements or open-close nodes
    # ... or comments I guess
    if any(x in line for x in validIndicators):
      lines.append(line)
    elif line == '':
      continue # useless space
    else:
      # just append to the previous line if it's a carryover
      addSpace = '' if (lines[-1].endswith(multiIndicators) or line in multiIndicators) else ' '
      lines[-1] += addSpace + line
  return lines

def getpotToInputTree(getpot):
  """
    Converts getpot input to RAVEN InputTree structure
    @ In, getpot, string, string to read in (with newlines, could be open file object)
    @ Out, tree, TreeStructure.NodeTree, tree with information
  """
  getpot = preprocessGetpot(getpot)
  roots = []
  parentNodes = [] # stack trace of parent nodes, as [First, Second, Third] nested
  currentNode = None # current node having objects added to it
  # track multiline vectors
  multilineKey = None       # attribute name
  multilineValue = None     # multiline value
  multilineIndicator = None # indicator to open/close, either ' or "
  for line in getpot:
    if multilineValue is not None:
      if multilineIndicator in line:
        # the multiline is done, so close and record it
        value, leftover = line.split(multilineIndicator, maxsplit=1)
        addSpace = ' ' if value.strip() else ''
        multilineValue += addSpace + value.strip() + multilineIndicator
        # set up to close entry
        attribute = multilineKey
        value = multilineValue
        closeEntry = True
        # reset
        multilineKey = None
        multilineValue = None
        multilineIndicator = None
      else:
        # still open and not closed, so keep appending
        addSpace = ' ' if multilineValue[-1] != multilineIndicator else ''
        multilineValue += addSpace + line.strip()
        closeEntry = False
    else:
      line = line.strip()
      #------------------
      # starting new node
      if line.startswith(('[./', '[')) and line.endswith(']') and line not in ('[../]','[]'):
        #if child node, stash the parent for now
        if currentNode is not None:
          parentNodes.append(currentNode)
        currentNode = TreeStructure.InputNode(tag=line.strip('[]./'))
        closeEntry = False

      #------------------
      # closing node
      elif line.startswith(('[../]', '[]')):
        if parentNodes:
          parentNodes[-1].append(currentNode)
          #make parent the active node
          currentNode = parentNodes.pop()
        else:
          #this is a root
          roots.append(currentNode)
          currentNode = None
        closeEntry = False
      #------------------
      # attributes and values
      elif '=' in line:
        attribute, value = (x.strip() for x in line.split('=', maxsplit=1))
        # TODO multilline, if "'" or '"' in line
        # ASSUME: both " and ' aren't used in the same line
        if any(x in line for x in multiIndicators):
          indicator = '\'' if '\'' in line else '\"'
          if line.count(indicator) % 2 == 0: # NOTE this may be more complex than needed, can you really use 4, 6?
            # closes within line
            value = value.strip('\'\"')
            closeEntry = True
          else:
            # multiline
            multilineKey = attribute
            multilineValue = value
            multilineIndicator = indicator
            closeEntry = False
            # DO NOT continue, keep going until multiline is closed
            continue
        else:
          # single line entry with no vector representation
          value = value.strip()
          closeEntry = True
      #------------------
      # we don't know what's going on here
      else:
        raise IOError('Unrecognized line syntax: "{}"'.format(line))
    #------------------
    # if the "value" if the "attribute" is closed, record the entry
    if closeEntry:
      if attribute in (c.tag for c in currentNode.children): #currentNode.attrib:
        raise IOError('Multiple entries defined with same name "{a}"!'.format(a=attribute))
      else:
        new = TreeStructure.InputNode(tag=attribute, text=value)
        currentNode.append(new)

  if multilineValue:
    raise IOError('There was a parsing error reading MOOSE input! Multiline attribute "{n}" opened by {i} but never closed!'
                  .format(i=multilineIndicator, n=multilineKey))

  return [TreeStructure.InputTree(root) for root in roots]

def writeGetpot(trees):
  """
    Renders multiple TreeStructure.InputTree instances as Getpot format.
    @ In, trees, list(TreeStructure.InputTree), structure to write
    @ Out, gps, string, rendering
  """
  gps = ''
  for tree in trees:
    gps += inputTreeToGetpot(tree)
  return gps

def inputTreeToGetpot(entry, rec=0):
  """
    Writes a InputTree to Getpot string.
    @ In, entry, InputStructure.InputTree or InputStructure.InputNode, structure
    @ In, rec, int, recursion depth
    @ Out, gp, string, getpot results
  """
  if isinstance(entry, TreeStructure.InputTree):
    root = entry.getroot()
  else:
    root = entry
  # if no sub information, just use tag and value
  if not(len(root) or root.attrib):
    val = root.text
    if not len(val):
      if rec == 0:
        start = '[{n}]\n'
        end = '[]\n'
      else:
        start = '  '*rec + '[./{n}]\n'
        end = '  '*rec + '[../]\n'
      gp =  start.format(n=root.tag)
      gp += end
    else:
      if ' ' in root.text:
        val =  '\"{}\"'.format(val)
      gp = '  '*rec + '{k} = {v}\n'.format(k=root.tag, v=val)
  else:
    if rec == 0:
      start = '[{n}]\n'
      end = '[]\n'
    else:
      start = '  '*rec + '[./{n}]\n'
      end = '  '*rec + '[../]\n'
    gp =  start.format(n=root.tag)
    # attributes
    for key, val in root.attrib.items():
      if ' ' in val:
        val = '\'{}\''.format(val)
      gp += '  '*(rec+1) + '{k} = {v}\n'.format(k=key, v=val)
    # nodes
    for node in root:
      gp += inputTreeToGetpot(node, rec=rec+1)
    gp += end
  return gp

def addNewPath(trees, target, value):
  """
    Adds a path and value to existing trees. Creates starting only at missing portion.
    @ In, trees, list(TreeStructure.InputTree), existing information
    @ In, target, list(string), nodal path to desired target as [Block1, Block2, entry]
    @ In, value, object, desired value for entry
    @ Ou, trees, list(TreeStructure.InputTree), modified list of trees
  """
  foundSoFar = None
  for t, targ in enumerate(target):
    found = findInGetpot(trees, target[:t+1])
    if found is not None:
      foundSoFar = found
    else:
      break
  # if starting from scratch ...
  if foundSoFar is None:
    base = TreeStructure.InputNode(target[0])
    tree = TreeStructure.InputTree(base)
    trees.append(tree)
    foundSoFar = [base]
    t = 1
  # add missing nodes
  current = foundSoFar[-1]
  for a, addTarg in enumerate(target[t:]):
    new = TreeStructure.InputNode(addTarg)
    current.append(new)
    if t + a == len(target) - 1:
      new.text = str(value)
    current = new
  return trees

def findInGetpot(trees, targetPath):
  """
    @ In, trees, list(TreeStructure.InputTrees), the tree(s) that get modified
    @ In, targetPath, list(string), target modification location as e.g. [Block1, Block2, entry]
    @ Out, found, TreeStructure.InputNode, found match (or None if not found)
  """
  targetRoot = targetPath[0]
  found = None
  for tree in trees:
    root = tree.getroot()
    if root.tag == targetRoot:
      # if just looking for root, return that
      if len(targetPath) == 1:
        return [root]
      found = findInTree(root, targetPath[1:])
      if found is not None:
        break
  return found

def findInTree(searchNode, targetPath, heritage=None):
  """
    Searches trees for element matching path given by target
    @ In, searchNode, TreeStructure.Inputnode, the node to search in for matches
    @ In, targetPath, list(string), target modification location as e.g. [Block1, Block2, entry]
    @ In, heritage, list(TreeStructure.InputNode), chain of objects leading to the match
    @ Out, found, list(TreeStructure.InputNode), found match and parents (or None if not found)
  """
  if heritage is None:
    heritage = []
  targetRoot = targetPath[0]
  found = None
  for sub in searchNode:
    if sub.tag == targetRoot:
      # if we're done searching and we have a match, return it
      if len(targetPath) == 1:
        return heritage + [sub]
      # otherwise keep searching for appropriate children
      found = findInTree(sub, targetPath[1:], heritage=heritage)
      if found:
        break
  return found
