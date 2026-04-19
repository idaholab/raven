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

def test_basis_enum_accepts_hosvd():
    xml = """<PostProcessor name='pp' subType='SparseSensing'>
      <Goal subType='reconstruction'>
        <features>X,T</features><target>T</target>
        <basis>HOSVD</basis><nModes>3</nModes><nSensors>3</nSensors>
        <optimizer>QR</optimizer><pivotParameter>time</pivotParameter>
      </Goal></PostProcessor>"""
    pp = SparseSensing()
    pp._handleInput(_parse(xml))
    assert pp.basis == 'HOSVD'

def test_legacy_randomprojetion_spelling_still_accepted():
    xml = """<PostProcessor name='pp' subType='SparseSensing'>
      <Goal subType='reconstruction'>
        <features>X,T</features><target>T</target>
        <basis>RandomProjetion</basis><nModes>2</nModes><nSensors>2</nSensors>
        <optimizer>QR</optimizer>
      </Goal></PostProcessor>"""
    pp = SparseSensing()
    pp._handleInput(_parse(xml))
    assert pp.basis.lower() == 'randomprojetion'

import numpy as np
import xarray as xr

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

def test_reshape_spatiotemporal_3d():
    pp = SparseSensing(); pp.reshape = 'spatiotemporal'
    X = np.arange(3*5*10, dtype=float).reshape(3, 5, 10)
    out = pp._reshapeForFit(X, pivotLen=5)
    assert out.shape == (3, 50)
    assert out[0, 11] == X[0, 1, 2]

def test_reshape_unknown_mode_raises():
    pp = SparseSensing(); pp.reshape = 'bogus'
    with pytest.raises(NotImplementedError, match="bogus"):
        pp._reshapeForFit(np.zeros((2, 3, 4)), pivotLen=3)

def _fake_inputIn_3d(nSamples=3, nTime=5, nSpace=20, seed=0):
    x = np.linspace(0, 1, nSpace)
    params = np.linspace(0.1, 1.0, nSamples)
    times = np.linspace(0, 1, nTime)
    # Separable manufactured field: T(x,t; p) = exp(-p*t)*sin(k*x)
    T = np.array([[np.exp(-p*t) * np.sin(3*np.pi*x) for t in times] for p in params])
    coords = {'RAVEN_sample_ID': np.arange(nSamples),
              'time': times,
              'pointID': np.arange(nSpace)}
    ds = xr.Dataset({'T': (('RAVEN_sample_ID','time','pointID'), T),
                     'X': (('RAVEN_sample_ID','pointID'), np.broadcast_to(x, (nSamples,nSpace)).copy())},
                    coords=coords)
    return {'Data': [(None, None, ds)]}

def test_run_snapshot_on_3d_input_returns_spatial_sensors():
    pp = SparseSensing()
    pp.name = 'spsl'
    pp.sensingFeatures = ['X','T']
    pp.sensingTarget = 'T'
    pp.basis = 'SVD'
    pp.nModes = 3
    pp.nSensors = 3
    pp.optimizer = 'QR'
    pp.pivotParameter = 'time'
    pp.reshape = 'snapshot'
    pp.seed = 42
    out = pp.run(_fake_inputIn_3d())
    assert out.sizes['sensor'] == 3
    assert 'time' not in out.dims

def test_run_spatiotemporal_on_3d_input_returns_schedule():
    pp = SparseSensing()
    pp.name = 'spsl'
    pp.sensingFeatures = ['X','T']
    pp.sensingTarget = 'T'
    pp.basis = 'SVD'
    pp.nModes = 3
    pp.nSensors = 3
    pp.optimizer = 'QR'
    pp.pivotParameter = 'time'
    pp.reshape = 'spatiotemporal'
    pp.seed = 42
    out = pp.run(_fake_inputIn_3d())
    assert out.sizes['sensor'] == 3
    assert 'time' in out.variables
    np.testing.assert_allclose(out['T'].data, np.exp(-0.1 * out['time'].data) * np.sin(3*np.pi*out['X'].data))

def test_run_raises_when_pivotParameter_not_in_input_dims():
    pp = SparseSensing()
    pp.name = 'spsl'
    pp.sensingFeatures = ['X', 'T']
    pp.sensingTarget = 'T'
    pp.basis = 'SVD'
    pp.nModes = 2
    pp.nSensors = 2
    pp.optimizer = 'QR'
    pp.pivotParameter = 'nonexistent'
    pp.reshape = 'snapshot'
    pp.seed = 0
    with pytest.raises(Exception, match="not found in input dims"):
        pp.run(_fake_inputIn_3d())

def test_run_raises_when_hosvd_without_pivotParameter():
    pp = SparseSensing()
    pp.name = 'spsl'
    pp.sensingFeatures = ['X', 'T']
    pp.sensingTarget = 'T'
    pp.basis = 'HOSVD'
    pp.nModes = 2
    pp.nSensors = 2
    pp.optimizer = 'QR'
    pp.pivotParameter = None
    pp.reshape = 'snapshot'
    pp.seed = 0
    with pytest.raises(Exception, match="HOSVD basis requires"):
        pp.run(_fake_inputIn_3d())

def test_hosvd_basis_rank_truncation():
    from ravenframework.Models.PostProcessors.SparseSensingBases import HOSVDBasis
    rng = np.random.RandomState(0)
    U1 = rng.randn(4, 2)
    U2 = rng.randn(5, 2)
    U3 = rng.randn(20, 3)
    core = rng.randn(2, 2, 3)
    X = np.einsum('ia,jb,kc,abc->ijk', U1, U2, U3, core) + 1e-6 * rng.randn(4, 5, 20)
    basis = HOSVDBasis(n_basis_modes=3).fit(X)
    assert basis.basis_matrix_.shape == (20, 3)
    np.testing.assert_allclose(basis.basis_matrix_.T @ basis.basis_matrix_, np.eye(3), atol=1e-9)

def test_hosvd_basis_rejects_2d_on_first_fit():
    from ravenframework.Models.PostProcessors.SparseSensingBases import HOSVDBasis
    with pytest.raises(ValueError, match="3-D"):
        HOSVDBasis(n_basis_modes=2).fit(np.zeros((4, 10)))

def test_hosvd_basis_ignores_2d_refit_when_prefit():
    from ravenframework.Models.PostProcessors.SparseSensingBases import HOSVDBasis
    rng = np.random.RandomState(0)
    X = rng.randn(3, 5, 20)
    basis = HOSVDBasis(n_basis_modes=3).fit(X)
    prev = basis.basis_matrix_.copy()
    basis.fit(X.reshape(15, 20))
    np.testing.assert_array_equal(basis.basis_matrix_, prev)

def test_hosvd_matrix_representation_truncates():
    from ravenframework.Models.PostProcessors.SparseSensingBases import HOSVDBasis
    rng = np.random.RandomState(0)
    basis = HOSVDBasis(n_basis_modes=3).fit(rng.randn(3, 5, 20))
    full = basis.matrix_representation()
    trunc = basis.matrix_representation(n_basis_modes=2)
    assert full.shape == (20, 3)
    assert trunc.shape == (20, 2)
    np.testing.assert_array_equal(trunc, full[:, :2])
