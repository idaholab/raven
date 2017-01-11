"""
  The UI module includes the different user interfaces available within RAVEN.

  Created on November 30, 2016
  @author: maljdp
"""

from __future__ import absolute_import

## These lines ensure that we do not have to do something like:
## 'from UI.Window import Window' outside
## of this submodule
from .ZoomableGraphicsView import ZoomableGraphicsView
from .BaseHierarchicalView import BaseHierarchicalView
from .DendrogramView import DendrogramView
from . import colors
from .FitnessView import FitnessView
from .ScatterView2D import ScatterView2D
from .ScatterView3D import ScatterView3D
from .SensitivityView import SensitivityView
from .TopologyMapView import TopologyMapView
from .HierarchyWindow import HierarchyWindow
from .TopologyWindow import TopologyWindow

## As these are not exposed to the user, we do not need a factory to dynamically
## allocate them. They will be explicitly called when needed everywhere in the
## code.
# from .Factory import knownTypes
# from .Factory import returnInstance
# from .Factory import returnClass

# We should not really need this as we do not use wildcard imports
__all__ = ['colors', 'HierarchylWindow', 'DendrogramView',
           'TopologyWindow', 'FitnessView', 'ScatterView2D',
           'ScatterView3D', 'SensitivityView', 'TopologyMapView']
