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

import numpy as np

def test_reshape_snapshot_2d_passthrough():
    # Current steady-state case: (samples, space) → unchanged
    pp = SparseSensing(); pp.reshape = 'snapshot'
    X = np.random.RandomState(0).randn(4, 100)
    out = pp._reshapeForFit(X, pivotLen=None)
    assert out.shape == (4, 100)
    np.testing.assert_array_equal(out, X)

def test_reshape_snapshot_3d_stacks_samples_and_time():
    # (samples=3, time=5, space=10) → (15, 10)
    pp = SparseSensing(); pp.reshape = 'snapshot'
    X = np.arange(3*5*10, dtype=float).reshape(3, 5, 10)
    out = pp._reshapeForFit(X, pivotLen=5)
    assert out.shape == (15, 10)
    # Row k·T + t should equal sample k at time t.
    np.testing.assert_array_equal(out[0], X[0, 0])
    np.testing.assert_array_equal(out[5], X[1, 0])
    np.testing.assert_array_equal(out[14], X[2, 4])
