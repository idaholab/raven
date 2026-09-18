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
# Matplotlib backend selection for the whole PlotInterfaces package.
# This MUST run before importing any plotting submodule: several of them (e.g.
# SamplePlot) import matplotlib.pyplot at module load, and the first pyplot import
# locks in the backend. Respect an explicit user choice (RAVEN_BACKEND / MPLBACKEND);
# otherwise fall back to the non-interactive Agg backend only when running headless
# (e.g. HPC/SSH with no X/Wayland display). This single, non-forcing selector is why
# the individual plot modules do not (and must not) call matplotlib.use(..., force=True),
# which would override the user's backend and GeneralPlot's interactive/screen destination.
import os as _os
import matplotlib as _matplotlib
_ravenBackend = _os.environ.get('RAVEN_BACKEND') or _os.environ.get('MPLBACKEND')
if _ravenBackend:
  _matplotlib.use(_ravenBackend)
elif not (_os.environ.get('DISPLAY') or _os.environ.get('WAYLAND_DISPLAY')):
  _matplotlib.use('Agg')

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
from .NSGAIIIReferenceDirectionPlot import NSGAIIIReferenceDirectionPlot
from .NSGAIIINichingHeatmapPlot import NSGAIIINichingHeatmapPlot
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
from .ThreeDVectorPlot import ThreeDVectorPlot
from .ThreeDTubePlot import ThreeDTubePlot
from .ThreeDConePlot import ThreeDConePlot
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
from .FeasibleRegionObjectiveContourPlot import FeasibleRegionObjectiveContourPlot
from .DecisionObjectiveMappingPlot import DecisionObjectiveMappingPlot
from .ParetoSurfacePlot import ParetoSurfacePlot
from .FeasibilityRadarPlot import FeasibilityRadarPlot
from .ParetoChartPlot import ParetoChartPlot
from .AdjustedEpsilonOptimalPlot import AdjustedEpsilonOptimalPlot
from .Factory import factory
