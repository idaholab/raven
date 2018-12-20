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
from __future__ import division, print_function, unicode_literals, absolute_import
import sys,os,re
import xml.etree.ElementTree as ET
import diffUtils as DU

from Tester import Differ

numTol = 1e-10 #effectively zero for our purposes

def findBranches(node,path,finished):
  """
    Iterative process to convert XML tree into list of entries
    @ In, node, ET.Element, whose children need sorting
    @ In, path, list(ET.Element), leading to node
    @ In, finished, list(list(ET.Element)), full entries
    @ Out, finished, list(list(ET.Element)), of full entries
  """
  for child in node:
    npath = path[:]+[child]
    if len(child)==0:
      finished.append(npath)
    else:
      finished = findBranches(child,npath,finished)
  return finished

def treeToList(node):
  """
    Converts XML tree to list of entries.  Useful to start recursive search.
    @ In, node, ET.Element, the xml tree root node to convert
    @ Out, treeToList, list(list(ET.Element)), of full paths to entries in xml tree
  """
  flattened = findBranches(node,[node],[])
  return list(tuple(f) for f in flattened)

def compareListEntry(aList,bList,**kwargs):
  """
    Comparse flattened XML entries for equality
    return bool is True if all tag, text, and attributes match, False otherwise
    return qual is percent of matching terms
    @ In, aList, list(ET.Element), first set
    @ In, bList, list(ET.Element), second set
    @ Out, compareListEntry, (bool,val), results
  """
  numMatch = 0       #number of matching points between entries
  totalMatchable = 0 #total tag, text, and attributes available to match
  match = True        #True if entries match
  diff = []           #tuple of (element, diff code, correct (a) value, test (b) value)
  options = kwargs
  for i in range(len(aList)):
    if i > len(bList) - 1:
      match = False
      diff.append((bList[-1],XMLDiff.missingChildNode,aList[i].tag,None))
      #could have matched the tag and attributes
      totalMatchable += 1 + len(aList[i].attrib.keys())
      #if text isn't empty, could have matched text, too
      if aList[i].text is not None and len(aList[i].text.strip())>0: totalMatchable+=1
      continue
    a = aList[i]
    b = bList[i]
    #match tag
    same,note = DU.compareStringsWithFloats(a.tag,b.tag,options["rel_err"], options["zero_threshold"], options["remove_whitespace"], options["remove_unicode_identifier"])
    totalMatchable += 1
    if not same:
      match = False
      diff.append((b,XMLDiff.notMatchTag,a.tag,b.tag))
    else:
      numMatch += 1
    #match text
    #if (a.text is None or len(a.text)>0) and (b.text is None or len(b.text)>0):
    same,note = DU.compareStringsWithFloats(a.text,b.text,options["rel_err"], options["zero_threshold"], options["remove_whitespace"], options["remove_unicode_identifier"])
    if not same:
      match = False
      diff.append((b,XMLDiff.notMatchText,str(a.text),str(b.text)))
      totalMatchable += 1
    else:
      if not(a.text is None or a.text.strip()!=''):
        numMatch += 1
        totalMatchable += 1
    #match attributes
    for attrib in a.attrib.keys():
      totalMatchable += 1
      if attrib not in b.attrib.keys():
        match = False
        diff.append((b,XMLDiff.missingAttribute,attrib,None))
        continue
      same,note = DU.compareStringsWithFloats(a.attrib[attrib],b.attrib[attrib],options["rel_err"], options["zero_threshold"], options["remove_whitespace"], options["remove_unicode_identifier"])
      if not same:
        match = False
        diff.append((b,XMLDiff.notMatchAttribute,(a,attrib),(b,attrib)))
      else:
        numMatch += 1
    #note attributes in b not in a
    for attrib in b.attrib.keys():
      if attrib not in a.attrib.keys():
        match = False
        diff.append((b,XMLDiff.extraAttribute,attrib,None))
        totalMatchable += 1
  # note elements in b not in a
  if len(bList) > len(aList):
    match = False
    for j in range(i,len(bList)):
      diff.append((aList[-1],XMLDiff.extraChildNode,bList[j].tag,None))
      #count tag and attributes
      totalMatchable += 1 + len(bList[j].attrib.keys())
      #if text isn't empty, count text, too
      if bList[i].text is not None and len(bList[i].text.strip())>0: totalMatchable+=1
  return (match,float(numMatch)/float(totalMatchable),diff)

def compareUnorderedElement(a,b,*args,**kwargs):
  """
    Compares two element trees and returns (same,message)
    where same is true if they are the same,
    and message is a list of the differences.
    Uses list of tree entries to find best match, instead of climbing the tree
    @ In, a, ET.Element, the first element
    @ In, b, ET.Element, the second element
    @ Out, compareUnorderedElement, (bool,[string]), results of comparison
  """
  same = True
  message = []
  options = kwargs
  matchvals = {}
  diffs = {}
  DU.setDefaultOptions(options)

  def failMessage(*args):
    """
      adds the fail message to the list
      @ In, args, list, The arguments to the fail message (will be converted with str())
      @ Out, failMessage, (bool,string), results
    """
    printArgs = []
    printArgs.extend(args)
    argsExpanded = " ".join([str(x) for x in printArgs])
    message.append(argsExpanded)
  if a.text != b.text:
    succeeded, note = DU.compareStringsWithFloats(a.text, b.text, options["rel_err"], options["zero_threshold"], options["remove_whitespace"], options["remove_unicode_identifier"])
    if not succeeded:
      same = False
      failMessage(note)
      return (same, message)
  aList = treeToList(a)
  bList = treeToList(b)
  #search a for matches in b
  for aEntry in aList:
    matchvals[aEntry] = {}
    diffs[aEntry] = {}
    for bEntry in bList:
      same,matchval,diff = compareListEntry(aEntry,bEntry,**options)
      if same:
        bList.remove(bEntry)
        del matchvals[aEntry]
        del diffs[aEntry]
        #since we found the match, remove from other near matches
        for closeKey in diffs.keys():
          if bEntry in diffs[closeKey].keys():
            del diffs[closeKey][bEntry]
            del matchvals[closeKey][bEntry]
        break
      else:
        matchvals[aEntry][bEntry] = matchval
        diffs[aEntry][bEntry] = diff
  if len(matchvals)==0: #all matches found
    return (True,'')
  else:
    note = ''
    for unmatched,close in matchvals.items():
      #print the path without a match
      note+='No match for '+'/'.join(list(m.tag for m in unmatched))+'\n'
      #print the tree of the nearest match
      note+='  Nearest unused match: '
      close = sorted(list(close.items()),key=lambda x:x[1],reverse=True)
      if len(close) > 1:
        closest = '/'.join(list(c.tag for c in close[0][0]))
      else:
        closest = '-none found-'
      note+='    '+ closest +'\n'
      #print what was different between them
      if len(close) > 1:
        diff =  diffs[unmatched][close[0][0]]
        for b,code,right,miss in diff:
          if b is None:
            b = str(b)
          if code is None:
            code = str(code)
          if right is None:
            right = str(right)
          if miss is None:
            miss = str(miss)
          if code == XMLDiff.missingChildNode:
            note+='    <'+b.tag+'> is missing child node: <'+right+'> vs <'+miss+'>\n'
          elif code == XMLDiff.missingAttribute:
            note+='    <'+b.tag+'> is missing attribute: "'+right+'"\n'
          elif code == XMLDiff.extraChildNode:
            note+='    <'+b.tag+'> has extra child node: <'+right+'>\n'
          elif code == XMLDiff.extraAttribute:
            note+='    <'+b.tag+'> has extra attribute: "'+right+'" = "'+b.attrib[right]+'"\n'
          elif code == XMLDiff.notMatchTag:
            note+='    <'+b.tag+'> tag does not match: <'+right+'> vs <'+miss+'>\n'
          elif code == XMLDiff.notMatchAttribute:
            note+='    <'+b.tag+'> attribute does not match: "'+right[1]+'" = "'+right[0].attrib[right[1]]+'" vs "'+miss[0].attrib[miss[1]]+'"\n'
          elif code == XMLDiff.notMatchText:
            note+='    <'+b.tag+'> text does not match: "'+right+'" vs "'+miss+'"\n'
          else:
            note+='     UNRECOGNIZED OPTION: "'+b.tag+'" "'+str(code)+'": "'+str(right)+'" vs "'+str(miss)+'"\n'

    return (False,[note])

def compareOrderedElement(a,b,*args,**kwargs):
  """
    Compares two element trees and returns (same,message) where same is true if they are the same, and message is a list of the differences
    @ In, a, ET.Element, the first element tree
    @ In, b, ET.Element, the second element tree
    @ In, args, dict, arguments
    @ In, kwargs, dict, keyword arguments
      accepted args:
        - none -
      accepted kwargs:
        path: a string to describe where the element trees are located (mainly
              used recursively)
    @ Out, compareOrderedElement, (bool,[string]), results of comparison
  """
  same = True
  message = []
  options = kwargs
  path = kwargs.get('path','')
  counter = kwargs.get('counter',0)
  DU.setDefaultOptions(options)

  def failMessage(*args):
    """
      adds the fail message to the list
      @ In, args, list, The arguments to the fail message (will be converted with str())
      @ Out, failMessage, (bool,string), results
    """
    printArgs = [path]
    printArgs.extend(args)
    argsExpanded = " ".join([str(x) for x in printArgs])
    message.append(argsExpanded)

  if a.tag != b.tag:
    same = False
    failMessage("mismatch tags ",a.tag,b.tag)
  else:
    path += a.tag + "/"
  if a.text != b.text:
    succeeded, note = DU.compareStringsWithFloats(a.text, b.text, options["rel_err"], options["zero_threshold"], options["remove_whitespace"], options["remove_unicode_identifier"])
    if not succeeded:
      same = False
      failMessage(note)
      return (same, message)
  differentKeys = set(a.keys()).symmetric_difference(set(b.keys()))
  sameKeys = set(a.keys()).intersection(set(b.keys()))
  if len(differentKeys) != 0:
    same = False
    failMessage("mismatch attribute keys ",differentKeys)
  for key in sameKeys:
    if a.attrib[key] != b.attrib[key]:
      same = False
      failMessage("mismatch attribute ",key,a.attrib[key],b.attrib[key])
  if len(a) != len(b):
    same = False
    failMessage("mismatch number of children ",len(a),len(b))
  else:
    if a.tag == b.tag:
      #find all matching XML paths
      #WARNING: this will mangle the XML, so other testing should happen above this!
      found=[]
      for i in range(len(a)):
        subOptions = dict(options)
        subOptions["path"] = path
        (sameChild,messageChild) = compareOrderedElement(a[i],b[i],*args,**subOptions)
        if sameChild: found.append((a[i],b[i]))
        same = same and sameChild
      #prune matches from trees
      for children in found:
        a.remove(children[0])
        b.remove(children[1])
      #once all pruning done, error on any remaining structure
      if counter==0: #on head now, recursion is finished
        if len(a)>0:
          aString = ET.tostring(a)
          if len(aString) > 80:
            message.append('Branches in gold not matching test...\n'+path)
          else:
            message.append('Branches in gold not matching test...\n'+path+
                           " "+aString)
        if len(b)>0:
          bString = ET.tostring(b)
          if len(bString) > 80:
            message.append('Branches in test not matching gold...\n'+path)
          else:
            message.append('Branches in test not matching gold...\n'+path+
                           " "+bString)
  return (same,message)

class XMLDiff:
  """
    XMLDiff is used for comparing xml files.
  """
  #static codes for differences
  missingChildNode  = 0
  missingAttribute   = 1
  extraChildNode    = 2
  extraAttribute     = 3
  notMatchTag       = 4
  notMatchAttribute = 5
  notMatchText      = 6

  def __init__(self, out_files, gold_files, **kwargs):
    """
      Create an XMLDiff class
      @ In, testDir, string, the directory where the test takes place
      @ In, out_files, List(string), the files to be compared.
      @ In, gold_files, List(String), the gold files to be compared.
      @ In, kwargs, dict,  other arguments that may be included:
            - 'unordered': indicates unordered sorting
      @ Out, None
    """
    assert len(out_files) == len(gold_files)
    self.__out_files = out_files
    self.__gold_files = gold_files
    self.__messages = ""
    self.__same = True
    self.__options = kwargs

  def diff(self):
    """
      Run the comparison.
      @ In, None
      @ Out, diff, (bool,string), (same,messages) where same is true if all the xml files are the same, and messages is a string with all the differences.
    """
    # read in files
    for testFilename, goldFilename in zip(self.__out_files, self.__gold_files):
      if not os.path.exists(testFilename):
        self.__same = False
        self.__messages += 'Test file does not exist: '+testFilename
      elif not os.path.exists(goldFilename):
        self.__same = False
        self.__messages += 'Gold file does not exist: '+goldFilename
      else:
        filesRead = True
        try:
          testRoot = ET.parse( testFilename ).getroot()
        except Exception as e:
          filesRead = False
          self.__messages += 'Exception reading file '+testFilename+': '+str(e.args)
        try:
          goldRoot = ET.parse( goldFilename ).getroot()
        except Exception as e:
          filesRead = False
          self.__messages += 'Exception reading file '+goldFilename+': '+str(e.args)
        if filesRead:
          if 'unordered' in self.__options.keys() and self.__options['unordered']:
            same,messages = compareUnorderedElement(goldRoot,testRoot,**self.__options)
          else:
            same,messages = compareOrderedElement(testRoot, goldRoot,**self.__options)
          if not same:
            self.__same = False
            separator = "\n"+" "*4
            self.__messages += "Mismatch between "+testFilename+" and "+goldFilename+separator
            self.__messages += separator.join(messages) + "\n"
        else:
          self.__same = False
    if '[' in self.__messages or ']' in self.__messages:
      self.__messages = self.__messages.replace('[','(')
      self.__messages = self.__messages.replace(']',')')
    return (self.__same,self.__messages)

class XML(Differ):
  """
  This is the class to use for handling the XML block.
  """
  @staticmethod
  def get_valid_params():
    params = Differ.get_valid_params()
    params.add_param('unordered', False, 'if true allow the tags in any order')
    params.add_param('zero_threshold',sys.float_info.min*4.0,'it represents the value below which a float is considered zero (XML comparison only)')
    params.add_param('remove_whitespace',False,'Removes whitespace before comparing xml node text if True')
    params.add_param('remove_unicode_identifier', False, 'if true, then remove u infront of a single quote')
    params.add_param('xmlopts','',"Options for xml checking")
    params.add_param('rel_err','','Relative Error for csv files or floats in xml ones')
    return params

  def __init__(self, name, params, test_dir):
    """
    Initializer for the class. Takes a String name and a dictionary params
    """
    Differ.__init__(self, name, params, test_dir)
    self.__xmlopts = {}
    if len(self.specs["rel_err"]) > 0:
      self.__xmlopts['rel_err'] = float(self.specs["rel_err"])
    self.__xmlopts['zero_threshold'] = float(self.specs["zero_threshold"])
    self.__xmlopts['unordered'     ] = bool(self.specs["unordered"])
    self.__xmlopts['remove_whitespace'] = self.specs['remove_whitespace'] == True
    self.__xmlopts['remove_unicode_identifier'] = self.specs['remove_unicode_identifier']
    if len(self.specs['xmlopts'])>0:
      self.__xmlopts['xmlopts'] = self.specs['xmlopts'].split(' ')

  def check_output(self):
    """
    Checks that the output matches the gold.
    returns (same, message) where same is true if the
    test passes, or false if the test failes.  message should
    gives a human readable explaination of the differences.
    """
    xml_files = self._get_test_files()
    gold_files = self._get_gold_files()
    xml_diff = XMLDiff(xml_files, gold_files, **self.__xmlopts)
    return xml_diff.diff()
