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
Factory for generating the instances of the  Models Module
"""

from ..EntityFactoryBase import EntityFactory

from ..CodeInterfaceBaseClass import CodeInterfaceBase
from .Generic.GenericCodeInterface import GenericCode
from .AccelerateCFD.AccelerateCFD import AcceleratedCFD
from .CobraTF.CTFinterface import CTF
from .Dymola.DymolaInterface import Dymola
from .MAAP5.MAAP5Interface import MAAP5
from .MAMMOTH.MAMMOTHInterface import MAMMOTH
from .MELCOR.melcorCombinedInterface import Melcor
from .MELCOR.melcorInterface import MelcorApp
from .MooseBasedApp.BisonAndMeshInterface import BisonAndMesh
from .MooseBasedApp.CubitInterface import Cubit
from .MooseBasedApp.CubitMooseInterface import CubitMoose
from .MooseBasedApp.MooseBasedAppInterface import MooseBasedApp
from .MooseBasedApp.BisonMeshScriptInterface import BisonMeshScript
from .Neutrino.neutrinoInterface import Neutrino
from .OpenModelica.OpenModelicaInterface import OpenModelica
from .PHISICS.PhisicsInterface import Phisics
from .PHISICS.PhisicsRelapInterface import PhisicsRelap5
from .Prescient.PrescientCodeInterface import Prescient
from .RAVEN.RAVENInterface import RAVEN
from .RELAP5.Relap5Interface import Relap5
from .RELAP5inssJp.Relap5inssJpInterface import Relap5inssJp
from .RELAP7.RELAP7Interface import RELAP7
from .Rattlesnake.RattlesnakeInterface import Rattlesnake
from .SCALE.ScaleInterface import Scale
from .SERPENT.SerpentInterface import SERPENT
from .SIMULATE3.SimulateInterface import Simulate
from .Saphire.SaphireInterface import Saphire
from .SIMULATE3.SimulateInterface import Simulate
from .WorkshopExamples.ProjectileInterface import Projectile
from .WorkshopExamples.ProjectileInterfaceNoCSV import ProjectileNoCSV
from .WorkshopExamples.BatemanInterface import BatemanSimple

factory = EntityFactory('Code', needsRunInfo=True)
factory.registerAllSubtypes(CodeInterfaceBase)
