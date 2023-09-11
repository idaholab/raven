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
  Factory interface for returning classes and instances from the
  Time Series Analysis module.
"""
from ..utils import InputData
from ..EntityFactoryBase import EntityFactory

from .TimeSeriesAnalyzer import TimeSeriesAnalyzer
from .Fourier import Fourier
from .ARMA import ARMA
from .VARMA import VARMA
from .MarkovAR import MarkovAR
from .Wavelet import Wavelet
from .PolynomialRegression import PolynomialRegression
from .RWD import RWD
from .STL import STL
from .Transformers import ZeroFilter, LogTransformer, ArcsinhTransformer, TanhTransformer, SigmoidTransformer, \
                          OutTruncation, MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, \
                          QuantileTransformer, Gaussianize, PreserveCDF, Differencing

factory = EntityFactory('TimeSeriesAnalyzer')
# TODO map lower case to upper case, because of silly ROM namespace problems
aliases = {'Fourier': 'fourier',
           'ARMA': 'arma',
           'VARMA': 'varma',
           'RWD': 'rwd',
           'Wavelet': 'wavelet',
           'ZeroFilter': 'zerofilter',
           'OutTruncation': 'outtruncation',
           'LogTransformer': 'logtransformer',
           'ArcsinhTransformer': 'arcsinhtransformer',
           'TanhTransformer': 'tanhtransformer',
           'SigmoidTransformer': 'sigmoidtransformer',
           'MaxAbsScaler': 'maxabsscaler',
           'MinMaxScaler': 'minmaxscaler',
           'StandardScaler': 'standardscaler',
           'RobustScaler': 'robustscaler',
           'QuantileTransformer': 'quantiletransformer',
           'Gaussianize': 'gaussianize',
           'PreserveCDF': 'preserveCDF',
           'Differencing': 'differencing'}
factory.registerAllSubtypes(TimeSeriesAnalyzer, alias=aliases)
