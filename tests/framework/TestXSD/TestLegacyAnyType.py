# Copyright 2026 Battelle Energy Alliance, LLC
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
Validate XSD generation for LegacyAnyType and simpleContent attributes.
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import os
import sys
import xml.etree.ElementTree as ET

raven_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(raven_dir)

from ravenframework.utils import InputData, InputTypes


def _fail(msg, fail_count):
  print("FAILED:", msg)
  return fail_count + 1


pass_fails = [0, 0]

# Build a spec that uses LegacyAnyType
LegacyNode = InputData.parameterInputFactory('LegacyNode',
                                             contentType=InputTypes.LegacyAnyType)

# Build a spec that uses simpleContent with attributes
SimpleNode = InputData.parameterInputFactory('SimpleNode',
                                             contentType=InputTypes.FloatType)
SimpleNode.addParam('units', param_type=InputTypes.StringType, required=False)

Outer = InputData.parameterInputFactory('Outer')
Outer.addSub(LegacyNode)
Outer.addSub(SimpleNode)

xsd_root = InputData.createXSD(Outer)
xsd_tree = ET.ElementTree(xsd_root)

elements = list(xsd_root.iter('xsd:element'))
legacy_elements = [elem for elem in elements if elem.get('name') == 'LegacyNode']
if not legacy_elements:
  pass_fails[1] = _fail("LegacyNode element not found in XSD", pass_fails[1])
else:
  legacy_type = legacy_elements[0].get('type')
  if legacy_type != InputTypes.LegacyAnyType.xmlType:
    pass_fails[1] = _fail("LegacyNode type is not xsd:anyType", pass_fails[1])
  else:
    pass_fails[0] += 1

# Validate that simpleContent attributes are attached to extension
simple_type_name = SimpleNode.__name__ + "_type"
simple_types = [ct for ct in xsd_root.iter('xsd:complexType') if ct.get('name') == simple_type_name]
if not simple_types:
  pass_fails[1] = _fail("SimpleNode complexType not found", pass_fails[1])
else:
  complex_type = simple_types[0]
  extensions = list(complex_type.iter('xsd:extension'))
  if not extensions:
    pass_fails[1] = _fail("SimpleNode extension not found", pass_fails[1])
  else:
    extension = extensions[0]
    extension_attrs = [a for a in extension if a.tag == 'xsd:attribute' and a.get('name') == 'units']
    if not extension_attrs:
      pass_fails[1] = _fail("SimpleNode attribute not found on extension", pass_fails[1])
    else:
      pass_fails[0] += 1
    direct_attrs = [a for a in complex_type if a.tag == 'xsd:attribute' and a.get('name') == 'units']
    if direct_attrs:
      pass_fails[1] = _fail("SimpleNode attribute incorrectly attached to complexType", pass_fails[1])
    else:
      pass_fails[0] += 1

print("passes", pass_fails[0], "fails", pass_fails[1])
sys.exit(pass_fails[1])

"""
  <TestInfo>
    <name>framework.test_xsd_legacy_any_type</name>
    <author>codex</author>
    <created>2026-02-09</created>
    <classesTested>utils.InputData, utils.InputTypes</classesTested>
    <description>
      Validates XSD generation for LegacyAnyType and for simpleContent attributes.
    </description>
    <revisions>
      <revision author="codex" date="2026-02-09">Initial version.</revision>
    </revisions>
  </TestInfo>
"""
