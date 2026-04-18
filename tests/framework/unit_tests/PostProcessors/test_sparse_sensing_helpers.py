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
  Unit tests for SparseSensing input-specification helpers (_handleInput parsing).
"""
import pytest
import xml.etree.ElementTree as ET
from ravenframework.Models.PostProcessors.SparseSensing import SparseSensing

def _parse(xml_str):
    spec = SparseSensing.getInputSpecification()()
    spec.parseNode(ET.fromstring(xml_str))
    return spec

def test_pivotParameter_is_parsed():
    xml = """<PostProcessor name='pp' subType='SparseSensing'>
      <Goal subType='reconstruction'>
        <features>X,Y,T</features><target>T</target>
        <basis>SVD</basis><nModes>2</nModes><nSensors>2</nSensors>
        <optimizer>QR</optimizer>
        <pivotParameter>time</pivotParameter>
        <reshape>snapshot</reshape>
      </Goal>
    </PostProcessor>"""
    pp = SparseSensing()
    pp._handleInput(_parse(xml))
    assert pp.pivotParameter == 'time'
    assert pp.reshape == 'snapshot'

def test_reshape_defaults_to_snapshot():
    xml = """<PostProcessor name='pp' subType='SparseSensing'>
      <Goal subType='reconstruction'>
        <features>X,Y,T</features><target>T</target>
        <basis>SVD</basis><nModes>2</nModes><nSensors>2</nSensors>
        <optimizer>QR</optimizer>
      </Goal>
    </PostProcessor>"""
    pp = SparseSensing()
    pp._handleInput(_parse(xml))
    assert pp.reshape == 'snapshot'
    assert pp.pivotParameter is None
