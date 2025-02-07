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
Created on 2016-Oct-21

@author: cogljj

This is used to test the xml differ
"""

import sys
import os
import unittest
import copy
import xml.etree.ElementTree as ET

scriptDir = os.path.dirname(os.path.abspath(__file__))
ravenDir = os.path.dirname(os.path.dirname(os.path.dirname(scriptDir)))

sys.path.append(os.path.join(ravenDir, "scripts", "TestHarness", "testers"))
sys.path.append(os.path.join(ravenDir, "rook"))

import XMLDiff


class TestXMLDiffer(unittest.TestCase):
  """ Unit testing for utility functions in rook/XMLDiff.py """

  def testCompareOrderedElement(self):
    """
    Tests the XMLDiff.compare_ordered_element function
    @ In, None
    @ Out, None
    """
    tree1 = ET.fromstring("<test></test>")
    same, _ = XMLDiff.compare_ordered_element(tree1, tree1)
    self.assertTrue(same, "simple")

    hello1 = ET.fromstring("<test>Hello  World</test>")
    hello2 = ET.fromstring("<test>Hello World</test>")
    same, _ = XMLDiff.compare_ordered_element(hello1, hello2)
    self.assertFalse(same, "whitespace")
    same, _ = XMLDiff.compare_ordered_element(hello1, hello2, remove_whitespace=True)
    self.assertTrue(same, "whitespace with remove")

  def testCompareUnorderedElement(self):
    """
    Tests the XMLDiff.compare_unordered_element function
    @ In, None
    @ Out, None
    """
    tree1 = ET.fromstring("<test></test>")
    tree2 = copy.deepcopy(tree1)
    same, _ = XMLDiff.compare_unordered_element(tree1, tree2)
    self.assertTrue(same, "simple")

    hello1 = ET.fromstring("<test>Hello  World</test>")
    hello2 = ET.fromstring("<test>Hello World</test>")
    same, _ = XMLDiff.compare_unordered_element(hello1, hello2)
    self.assertFalse(same, "whitespace")
    same, _ = XMLDiff.compare_unordered_element(hello1, hello2, remove_whitespace=True)
    self.assertTrue(same, "whitespace with remove")

  def testIgnoreSubnodesFromTree(self):
    """
    Tests the XMLDiff.ignore_subnodes_from_tree function
    @ In, None
    @ Out, None
    """
    tree1 = ET.fromstring("<test> <a>0</a> <b>0</b> </test>")
    tree2 = copy.deepcopy(tree1)
    aTree, bTree, success = XMLDiff.ignore_subnodes_from_tree(tree1, tree2, ignored_nodes=["./b"])
    self.assertTrue(success, "ignoring child node in both trees")
    same, _ = XMLDiff.compare_unordered_element(aTree, bTree)
    self.assertTrue(same, "comparing roots after ignoring child node in both trees")

    treeAttrib1 = ET.fromstring("<test><a n='0'>1</a><a n='1'/></test>")
    treeAttrib2 = ET.fromstring("<test><a n='0'>2</a><a n='1'/></test>")
    aTree, bTree, success = XMLDiff.ignore_subnodes_from_tree(treeAttrib1, treeAttrib2, ignored_nodes=["./a[@n='0']"])
    self.assertTrue(success, "ignoring child node in both trees via attribute")
    same, _ = XMLDiff.compare_unordered_element(aTree, bTree)
    self.assertTrue(same, "comparing roots after ignoring child node in both trees via attribute")

    treeExclusive1 = ET.fromstring("<test> <a/> <c/> </test>")
    treeExclusive2 = ET.fromstring("<test> <a/> <b/> </test>")
    aTree, bTree, success = XMLDiff.ignore_subnodes_from_tree(treeExclusive1, treeExclusive2, ignored_nodes=["./b", "./c"])
    self.assertTrue(success, "ignoring multiple nodes exclusive to trees")
    same, _ = XMLDiff.compare_unordered_element(aTree, bTree)
    self.assertTrue(same, "comparing roots after ignoring multiple nodes exclusive to trees")

  def testComparePathsInSubnodes(self):
    """
    Tests the XMLDiff.compare_paths_in_subnodes function
    @ In, None
    @ Out, None
    """
    # exactly equal paths
    path1 = os.path.sep.join(["..", "some", "rel", "path"])
    path2 = copy.deepcopy(path1)
    node1 = ET.fromstring(f"<root><test>{path1}</test></root>")
    node2 = ET.fromstring(f"<root><test>{path2}</test></root>")
    _, _, equal = XMLDiff.compare_paths_in_subnodes(node1, node2, ["./test"])
    self.assertTrue(equal, "comparing equal paths")

    # this should fail
    path1 = "foo"
    path2 = "bar"
    node1 = ET.fromstring(f"<root><test>{path1}</test></root>")
    node2 = ET.fromstring(f"<root><test>{path2}</test></root>")
    _, _, equal = XMLDiff.compare_paths_in_subnodes(node1, node2, ["./test"])
    self.assertFalse(equal, "comparing different paths")

    # looking at multiple nodes with paths
    path1 = "foo"
    path2 = "bar"
    node1 = ET.fromstring(f"<root><test>{path1}</test><other>{path2}</other></root>")
    node2 = ET.fromstring(f"<root><test>{path1}</test><other>{path2}</other></root>")
    _, _, equal = XMLDiff.compare_paths_in_subnodes(node1, node2, ["./test", "./other"])
    self.assertTrue(equal, "comparing multiple paths that match")

    # looking at multiple nodes with one path that fails
    path1 = "foo"
    path2 = "bar"
    node1 = ET.fromstring(f"<root><test>{path1}</test><other>{path1}</other></root>")
    node2 = ET.fromstring(f"<root><test>{path1}</test><other>{path2}</other></root>")
    _, _, equal = XMLDiff.compare_paths_in_subnodes(node1, node2, ["./test", "./other"])
    self.assertFalse(equal, "comparing multiple paths with one mismatch")

    # Compare between Windows paths and POSIX paths
    path_parts = ["some", "path"]
    posix = "/".join(path_parts)
    windows = "\\".join(path_parts)
    node1 = ET.fromstring(f"<root><test>{posix}</test></root>")
    node2 = ET.fromstring(f"<root><test>{windows}</test></root>")
    _, _, equal = XMLDiff.compare_paths_in_subnodes(node1, node2, ["./test"])
    self.assertTrue(equal, "comparing Windows and POSIX paths")


"""
  <TestInfo>
    <name>framework.test_xml_differ</name>
    <author>cogljj</author>
    <created>2016-10-21</created>
    <classesTested>None</classesTested>
    <description>
       This test is aimed to check the xml differ program
    </description>
    <revisions>
      <revision author="alfoa" date="2017-01-21">Adding this test description.</revision>
      <revision author="sotogj" date="2023-09-07">Adding tests for ignoring nodes.</revision>
      <revision author="j-bryan" date="2024-01-09">
        Refactor to use unittest and adding tests for file path node parsing
      </revision>
    </revisions>
  </TestInfo>
"""
