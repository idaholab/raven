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
  Created on April 1, 2021

  @author: talbpaul
"""

from ...EntityFactoryBase import EntityFactory

# Entities
from .PlotInterface import PlotInterface
from .SamplePlot import SamplePlot
from .GeneralPlot import GeneralPlot
from .OptPath import OptPath
from .PopulationPlot import PopulationPlot
from .SyntheticCloud import SyntheticCloud
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
from .CoreLayoutPlot import CoreLayoutPlot
from .ShuffleArrowsPlot import ShufflingSchemePlot

factory = EntityFactory('Plot')
factory.registerType('GeneralPlot', GeneralPlot)
factory.registerType('SamplePlot', SamplePlot)
factory.registerType('OptPath', OptPath)
factory.registerType('SyntheticCloud', SyntheticCloud)
factory.registerType('PopulationPlot', PopulationPlot)
factory.registerType('OptParallelCoordinatePlot', OptParallelCoordinatePlot)
factory.registerType('NSGAParetoFrontPlot', NSGAParetoFrontPlot)
factory.registerType('NSGAFrontAnimation', NSGAFrontAnimation)
factory.registerType('NSGARankHistoryPlot', NSGARankHistoryPlot)
factory.registerType('NSGACrowdingDistancePlot', NSGACrowdingDistancePlot)
factory.registerType('NSGAFrontRankAnimation', NSGAFrontRankAnimation)
factory.registerType('NSGAIIIReferenceDirectionPlot', NSGAIIIReferenceDirectionPlot)
factory.registerType('NSGAIIINichingHeatmapPlot', NSGAIIINichingHeatmapPlot)
factory.registerType('ObjectiveContourAnimationPlot', ObjectiveContourAnimationPlot)
factory.registerType('ParetoDiagnosticsPlot', ParetoDiagnosticsPlot)
factory.registerType('DominanceHeatMapPlot', DominanceHeatMapPlot)
factory.registerType('TradeoffSlicePlot', TradeoffSlicePlot)
factory.registerType('HypervolumeMoviePlot', HypervolumeMoviePlot)
factory.registerType('ConstraintActivityTimelinePlot', ConstraintActivityTimelinePlot)
factory.registerType('DiversityRadarPlot', DiversityRadarPlot)
factory.registerType('FitnessFunnelPlot', FitnessFunnelPlot)
factory.registerType('SamplingCoverageMapPlot', SamplingCoverageMapPlot)
factory.registerType('BubbleTradeoffPlot', BubbleTradeoffPlot)
factory.registerType('ThreeDVectorPlot', ThreeDVectorPlot)
factory.registerType('ThreeDTubePlot', ThreeDTubePlot)
factory.registerType('ThreeDConePlot', ThreeDConePlot)
factory.registerType('AttainmentSurfacePlot', AttainmentSurfacePlot)
factory.registerType('ConstraintViolationHeatmapPlot', ConstraintViolationHeatmapPlot)
factory.registerType('ResponseSurfaceOverlayPlot', ResponseSurfaceOverlayPlot)
factory.registerType('MultiRunUncertaintyPlot', MultiRunUncertaintyPlot)
factory.registerType('RadvizEmbeddingPlot', RadvizEmbeddingPlot)
factory.registerType('ProsectionMatrixPlot', ProsectionMatrixPlot)
factory.registerType('PreferenceSweepAnimationPlot', PreferenceSweepAnimationPlot)
factory.registerType('StarCoordinatesPlot', StarCoordinatesPlot)
factory.registerType('SelfOrganizingMapPlot', SelfOrganizingMapPlot)
factory.registerType('ChordDiagramPlot', ChordDiagramPlot)
factory.registerType('GlyphRadarPlot', GlyphRadarPlot)
factory.registerType('CoreLayoutPlot', CoreLayoutPlot)
factory.registerType('ShufflingSchemePlot', ShufflingSchemePlot)
