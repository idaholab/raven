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
  Implementations of plottings strategies for OutStreams.

  Created April 1, 2021
  @author: talbpaul
"""
from .PlotInterface import PlotInterface
from .SamplePlot import SamplePlot
from .GeneralPlot import GeneralPlot as Plot
from .OptPath import OptPath
from .SyntheticCloud import SyntheticCloud
from .PopulationPlot import PopulationPlot
from .OptParallelCoordinate import OptParallelCoordinatePlot
from .NSGAParetoFrontPlot import NSGAParetoFrontPlot
from .NSGAFrontAnimation import NSGAFrontAnimation
from .NSGARankHistoryPlot import NSGARankHistoryPlot
from .NSGACrowdingDistancePlot import NSGACrowdingDistancePlot
from .NSGAFrontRankAnimation import NSGAFrontRankAnimation
from .ObjectiveContourAnimation import ObjectiveContourAnimationPlot
from .ParetoDiagnosticsPlot import ParetoDiagnosticsPlot
from .DominanceHeatMapPlot import DominanceHeatMapPlot
from .TradeoffSlicePlot import TradeoffSlicePlot
from .HypervolumeMoviePlot import HypervolumeMoviePlot
from .ConstraintActivityTimelinePlot import ConstraintActivityTimelinePlot
from .DiversityRadarPlot import DiversityRadarPlot
from .FitnessFunnelPlot import FitnessFunnelPlot
from .SamplingCoverageMapPlot import SamplingCoverageMapPlot
from .BubbleTradeoffPlot import BubbleTradeoffPlot
from .AttainmentSurfacePlot import AttainmentSurfacePlot
from .ConstraintViolationHeatmapPlot import ConstraintViolationHeatmapPlot
from .ResponseSurfaceOverlayPlot import ResponseSurfaceOverlayPlot
from .MultiRunUncertaintyPlot import MultiRunUncertaintyPlot
from .RadvizEmbeddingPlot import RadvizEmbeddingPlot
from .ProsectionMatrixPlot import ProsectionMatrixPlot
from .PreferenceSweepAnimationPlot import PreferenceSweepAnimationPlot
from .StarCoordinatesPlot import StarCoordinatesPlot
from .SelfOrganizingMapPlot import SelfOrganizingMapPlot
from .ChordDiagramPlot import ChordDiagramPlot
from .GlyphRadarPlot import GlyphRadarPlot
from .CoreLayoutPlot import CoreLayoutPlot
from .Factory import factory
