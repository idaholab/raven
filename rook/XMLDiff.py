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
This implements a test to compare two XML files.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import os
import xml.etree.ElementTree as ET
import DiffUtils as DU

from Tester import Differ

numTol = 1e-10 #effectively zero for our purposes

def find_branches(node, path, finished):
  """
    Iterative process to convert XML tree into list of entries
    @ In, node, ET.Element, whose children need sorting
    @ In, path, list(ET.Element), leading to node
    @ In, finished, list(list(ET.Element)), full entries
    @ Out, finished, list(list(ET.Element)), of full entries
  """
  for child in node:
    npath = path[:]+[child]
    if len(child) == 0:
      finished.append(npath)
    else:
      finished = find_branches(child, npath, finished)
  return finished

def tree_to_list(node):
  """
    Converts XML tree to list of entries.  Useful to start recursive search.
    @ In, node, ET.Element, the xml tree root node to convert
    @ Out, tree_to_list, list(list(ET.Element)), of full paths to entries in xml tree
  """
  flattened = find_branches(node, [node], [])
  return list(tuple(f) for f in flattened)

def compare_list_entry(a_list, b_list, **kwargs):
  """
    Comparse flattened XML entries for equality
    return bool is True if all tag, text, and attributes match, False otherwise
    return qual is percent of matching terms
    @ In, a_list, list(ET.Element), first set
    @ In, b_list, list(ET.Element), second set
    @ Out, compare_list_entry, (bool,val), results
  """
  num_match = 0       #number of matching points between entries
  total_matchable = 0 #total tag, text, and attributes available to match
  match = True        #True if entries match
  diff = []           #tuple of (element, diff code, correct (a) value, test (b) value)
  options = kwargs
  for i in range(len(a_list)):
    if i > len(b_list) - 1:
      match = False
      diff.append((b_list[-1], XMLDiff.missingChildNode, a_list[i].tag, None))
      #could have matched the tag and attributes
      total_matchable += 1 + len(a_list[i].attrib.keys())
      #if text isn't empty, could have matched text, too
      if a_list[i].text is not None and len(a_list[i].text.strip()) > 0:
        total_matchable += 1
      continue
    a_item = a_list[i]
    b_item = b_list[i]
    #match tag
    same, _ = DU.compare_strings_with_floats(a_item.tag, b_item.tag,
                                             options["rel_err"],
                                             options["zero_threshold"],
                                             options["remove_whitespace"],
                                             options["remove_unicode_identifier"])
    total_matchable += 1
    if not same:
      match = False
      diff.append((b_item, XMLDiff.notMatchTag, a_item.tag, b_item.tag))
    else:
      num_match += 1
    #match text
    #if (a_item.text is None or len(a_item.text)>0) and (b_item.text is None or len(b_item.text)>0):
    same, _ = DU.compare_strings_with_floats(a_item.text,
                                             b_item.text,
                                             options["rel_err"],
                                             options["zero_threshold"],
                                             options["remove_whitespace"],
                                             options["remove_unicode_identifier"])
    if not same:
      match = False
      diff.append((b_item, XMLDiff.notMatchText, str(a_item.text), str(b_item.text)))
      total_matchable += 1
    else:
      if not(a_item.text is None or a_item.text.strip() != ''):
        num_match += 1
        total_matchable += 1
    #match attributes
    for attrib in a_item.attrib.keys():
      total_matchable += 1
      if attrib not in b_item.attrib.keys():
        match = False
        diff.append((b_item, XMLDiff.missingAttribute, attrib, None))
        continue
      same, _ = DU.compare_strings_with_floats(a_item.attrib[attrib],
                                               b_item.attrib[attrib],
                                               options["rel_err"],
                                               options["zero_threshold"],
                                               options["remove_whitespace"],
                                               options["remove_unicode_identifier"])
      if not same:
        match = False
        diff.append((b_item, XMLDiff.notMatchAttribute, (a_item, attrib), (b_item, attrib)))
      else:
        num_match += 1
    #note attributes in b_item not in a_item
    for attrib in b_item.attrib.keys():
      if attrib not in a_item.attrib.keys():
        match = False
        diff.append((b_item, XMLDiff.extraAttribute, attrib, None))
        total_matchable += 1
  # note elements in b not in a
  if len(b_list) > len(a_list):
    match = False
    i = len(a_list) - 1
    for j in range(i, len(b_list)):
      diff.append((a_list[-1], XMLDiff.extraChildNode, b_list[j].tag, None))
      #count tag and attributes
      total_matchable += 1 + len(b_list[j].attrib.keys())
      #if text isn't empty, count text, too
      if b_list[i].text is not None and len(b_list[i].text.strip()) > 0:
        total_matchable += 1
  return (match, float(num_match)/float(total_matchable), diff)

def compare_unordered_element(a_element, b_element, **kwargs):
  """
    Compares two element trees and returns (same,message)
    where same is true if they are the same,
    and message is a list of the differences.
    Uses list of tree entries to find best match, instead of climbing the tree
    @ In, a_element, ET.Element, the first element
    @ In, b_element, ET.Element, the second element
    @ Out, compare_unordered_element, (bool,[string]), results of comparison
  """
  same = True
  message = []
  options = kwargs
  matchvals = {}
  diffs = {}
  DU.set_default_options(options)

  def fail_message(*args):
    """
      adds the fail message to the list
      @ In, args, list, The arguments to the fail message (will be converted with str())
      @ Out, fail_message, (bool,string), results
    """
    print_args = []
    print_args.extend(args)
    args_expanded = " ".join([str(x) for x in print_args])
    message.append(args_expanded)
  if a_element.text != b_element.text:
    succeeded, note = DU.compare_strings_with_floats(a_element.text,
                                                     b_element.text,
                                                     options["rel_err"],
                                                     options["zero_threshold"],
                                                     options["remove_whitespace"],
                                                     options["remove_unicode_identifier"])
    if not succeeded:
      same = False
      fail_message(note)
      return (same, message)
  a_list = tree_to_list(a_element)
  b_list = tree_to_list(b_element)
  #search a for matches in b
  for a_entry in a_list:
    matchvals[a_entry] = {}
    diffs[a_entry] = {}
    for b_entry in b_list:
      same, matchval, diff = compare_list_entry(a_entry, b_entry, **options)
      if same:
        b_list.remove(b_entry)
        del matchvals[a_entry]
        del diffs[a_entry]
        #since we found the match, remove from other near matches
        for close_key in diffs:
          if b_entry in diffs[close_key].keys():
            del diffs[close_key][b_entry]
            del matchvals[close_key][b_entry]
        break
      else:
        matchvals[a_entry][b_entry] = matchval
        diffs[a_entry][b_entry] = diff
  if len(matchvals) == 0: #all matches found
    return (True, '')
  note = ''
  for unmatched, close in matchvals.items():
    #print the path without a match
    path = '/'.join(list(m.tag for m in unmatched))
    note += 'No match for gold node {}\n'.format(path)
    note += '               tag: {}\n'.format(unmatched[-1].tag)
    note += '              attr: {}\n'.format(unmatched[-1].attrib)
    note += '              text: {}\n'.format(unmatched[-1].text)
    #print the tree of the nearest match
    note += '  Nearest unused match: '
    close = sorted(list(close.items()), key=lambda x: x[1], reverse=True)
    if len(close) > 1:
      closest = '/'.join(list(c.tag for c in close[0][0]))
    else:
      closest = '-none found-'
    note += '    '+ closest +'\n'
    #print what was different between them
    if len(close) > 1:
      diff = diffs[unmatched][close[0][0]]
      for b_diff, code, right, miss in diff:
        if b_diff is None:
          b_diff = str(b_diff)
        if code is None:
          code = str(code)
        if right is None:
          right = str(right)
        if miss is None:
          miss = str(miss)
        if code == XMLDiff.missingChildNode:
          note += '    <'+b_diff.tag+'> is missing child node: <'+right+'> vs <'+miss+'>\n'
        elif code == XMLDiff.missingAttribute:
          note += '    <'+b_diff.tag+'> is missing attribute: "'+right+'"\n'
        elif code == XMLDiff.extraChildNode:
          note += '    <'+b_diff.tag+'> has extra child node: <'+right+'>\n'
        elif code == XMLDiff.extraAttribute:
          note += '    <'+b_diff.tag+'> has extra attribute: "'+right+\
            '" = "'+b_diff.attrib[right]+'"\n'
        elif code == XMLDiff.notMatchTag:
          note += '    <'+b_diff.tag+'> tag does not match: <'+right+'> vs <'+miss+'>\n'
        elif code == XMLDiff.notMatchAttribute:
          note += '    <'+b_diff.tag+'> attribute does not match: "'+right[1]+\
            '" = "'+right[0].attrib[right[1]]+'" vs "'+miss[0].attrib[miss[1]]+'"\n'
        elif code == XMLDiff.notMatchText:
          note += '    <'+b_diff.tag+'> text does not match: "'+right+'" vs "'+miss+'"\n'
        else:
          note += '     UNRECOGNIZED OPTION: "'+b_diff.tag+'" "'+str(code)+\
            '": "'+str(right)+'" vs "'+str(miss)+'"\n'

  return (False, [note])

def compare_ordered_element(a_element, b_element, *args, **kwargs):
  """
    Compares two element trees and returns (same,message) where same is true
      if they are the same, and message is a list of the differences
    @ In, a_element, ET.Element, the first element tree
    @ In, b_element, ET.Element, the second element tree
    @ In, args, dict, arguments
    @ In, kwargs, dict, keyword arguments
      accepted args:
        - none -
      accepted kwargs:
        path: a string to describe where the element trees are located (mainly
              used recursively)
    @ Out, compare_ordered_element, (bool,[string]), results of comparison
  """
  same = True
  message = []
  options = kwargs
  path = kwargs.get('path', '')
  counter = kwargs.get('counter', 0)
  DU.set_default_options(options)

  def fail_message(*args):
    """
      adds the fail message to the list
      @ In, args, list, The arguments to the fail message (will be converted with str())
      @ Out, fail_message, (bool,string), results
    """
    print_args = [path]
    print_args.extend(args)
    args_expanded = " ".join([str(x) for x in print_args])
    message.append(args_expanded)

  if a_element.tag != b_element.tag:
    same = False
    fail_message("mismatch tags ", a_element.tag, b_element.tag)
  else:
    path += a_element.tag + "/"
  if a_element.text != b_element.text:
    succeeded, note = DU.compare_strings_with_floats(a_element.text,
                                                     b_element.text,
                                                     options["rel_err"],
                                                     options["zero_threshold"],
                                                     options["remove_whitespace"],
                                                     options["remove_unicode_identifier"])
    if not succeeded:
      same = False
      fail_message(note)
      return (same, message)
  different_keys = set(a_element.keys()).symmetric_difference(set(b_element.keys()))
  same_keys = set(a_element.keys()).intersection(set(b_element.keys()))
  if len(different_keys) != 0:
    same = False
    fail_message("mismatch attribute keys ", different_keys)
  for key in same_keys:
    if a_element.attrib[key] != b_element.attrib[key]:
      same = False
      fail_message("mismatch attribute ", key, a_element.attrib[key], b_element.attrib[key])
  if len(a_element) != len(b_element):
    same = False
    fail_message("mismatch number of children ", len(a_element), len(b_element))
  else:
    if a_element.tag == b_element.tag:
      #find all matching XML paths
      #WARNING: this will mangle the XML, so other testing should happen above this!
      found = []
      for i in range(len(a_element)):
        sub_options = dict(options)
        sub_options["path"] = path
        (same_child, _) = compare_ordered_element(a_element[i], b_element[i], *args, **sub_options)
        if same_child:
          found.append((a_element[i], b_element[i]))
        same = same and same_child
      #prune matches from trees
      for children in found:
        a_element.remove(children[0])
        b_element.remove(children[1])
      #once all pruning done, error on any remaining structure
      if counter == 0: #on head now, recursion is finished
        if len(a_element) > 0:
          a_string = ET.tostring(a_element)
          if len(a_string) > 80:
            message.append('Branches in gold not matching test...\n'+path)
          else:
            message.append('Branches in gold not matching test...\n'+path+
                           " "+a_string)
        if len(b_element) > 0:
          b_string = ET.tostring(b_element)
          if len(b_string) > 80:
            message.append('Branches in test not matching gold...\n'+path)
          else:
            message.append('Branches in test not matching gold...\n'+path+
                           " "+b_string)
  return (same, message)

class XMLDiff:
  """
    XMLDiff is used for comparing xml files.
  """
  #static codes for differences
  missingChildNode = 0
  missingAttribute = 1
  extraChildNode = 2
  extraAttribute = 3
  notMatchTag = 4
  notMatchAttribute = 5
  notMatchText = 6

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
      @ Out, diff, (bool,string), (same,messages) where same is true if all
          the xml files are the same, and messages is a string with all the
          differences.
    """
    # read in files
    for test_filename, gold_filename in zip(self.__out_files, self.__gold_files):
      if not os.path.exists(test_filename):
        self.__same = False
        self.__messages += 'Test file does not exist: '+test_filename
      elif not os.path.exists(gold_filename):
        self.__same = False
        self.__messages += 'Gold file does not exist: '+gold_filename
      else:
        files_read = True
        try:
          test_root = ET.parse(test_filename).getroot()
        except Exception as exp:
          files_read = False
          self.__messages += 'Exception reading file '+test_filename+': '+str(exp.args)
        try:
          gold_root = ET.parse(gold_filename).getroot()
        except Exception as exp:
          files_read = False
          self.__messages += 'Exception reading file '+gold_filename+': '+str(exp.args)
        if files_read:
          if 'unordered' in self.__options.keys() and self.__options['unordered']:
            same, messages = compare_unordered_element(gold_root, test_root, **self.__options)
          else:
            same, messages = compare_ordered_element(test_root, gold_root, **self.__options)
          if not same:
            self.__same = False
            separator = "\n"+" "*4
            self.__messages += "Mismatch between "+test_filename+" and "+gold_filename+separator
            self.__messages += separator.join(messages) + "\n"
        else:
          self.__same = False
    if '[' in self.__messages or ']' in self.__messages:
      self.__messages = self.__messages.replace('[', '(')
      self.__messages = self.__messages.replace(']', ')')
    return (self.__same, self.__messages)

class XML(Differ):
  """
  This is the class to use for handling the XML block.
  """
  @staticmethod
  def get_valid_params():
    """
      Return the valid parameters for this class.
      @ In, None
      @ Out, params, _ValidParameters, return the parameters.
    """
    params = Differ.get_valid_params()
    params.add_param('unordered', False, 'if true allow the tags in any order')
    params.add_param('zero_threshold', sys.float_info.min*4.0, 'it represents '
                     +'the value below which a float is considered zero (XML comparison only)')
    params.add_param('remove_whitespace', False,
                     'Removes whitespace before comparing xml node text if True')
    params.add_param('remove_unicode_identifier', False,
                     'if true, then remove u infront of a single quote')
    params.add_param('xmlopts', '', "Options for xml checking")
    params.add_param('rel_err', '', 'Relative Error for csv files or floats in xml ones')
    return params

  def __init__(self, name, params, test_dir):
    """
      Initializer for the class. Takes a String name and a dictionary params
      @ In, name, string, name of the test.
      @ In, params, dictionary, parameters for the class
      @ In, test_dir, string, path to the test.
      @ Out, None.
    """
    Differ.__init__(self, name, params, test_dir)
    self.__xmlopts = {}
    if len(self.specs["rel_err"]) > 0:
      self.__xmlopts['rel_err'] = float(self.specs["rel_err"])
    self.__xmlopts['zero_threshold'] = float(self.specs["zero_threshold"])
    self.__xmlopts['unordered'] = bool(self.specs["unordered"])
    self.__xmlopts['remove_whitespace'] = bool(self.specs['remove_whitespace'])
    self.__xmlopts['remove_unicode_identifier'] = self.specs['remove_unicode_identifier']
    if len(self.specs['xmlopts']) > 0:
      self.__xmlopts['xmlopts'] = self.specs['xmlopts'].split(' ')

  def check_output(self):
    """
      Checks that the output matches the gold.
      returns (same, message) where same is true if the
      test passes, or false if the test failes.  message should
      gives a human readable explaination of the differences.
      @ In, None
      @ Out, (same, message), same is true if the tests passes.
    """
    xml_files = self._get_test_files()
    gold_files = self._get_gold_files()
    xml_diff = XMLDiff(xml_files, gold_files, **self.__xmlopts)
    return xml_diff.diff()
