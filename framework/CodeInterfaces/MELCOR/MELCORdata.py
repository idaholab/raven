# Copyright 2017 University of Rome La Sapienza and Battelle Energy Alliance, LLC
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
  Created on April 18, 2017
  @author: Matteo Donorio (University of Rome La Sapienza),
           Fabio Gianneti (University of Rome La Sapienza),
           Andrea Alfonsi (INL)
"""
from __future__ import division, print_function, unicode_literals, absolute_import
from melcorTools import MCRBin
import pandas as pd
import warnings
warnings.simplefilter('default',DeprecationWarning)
import os
import re
import copy
import itertools
import fileinput
from BaseClasses import BaseType
from CodeInterfaceBaseClass import CodeInterfaceBase
import melcorCombinedInterface

class MELCORdata():
  """
    This class is the CodeInterface for MELGEN (a sub-module of Melcor)
  """

  def __init__(self,origInputFiles):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    self.VarSrch = melcorCombinedInterface.MelcorApp.VarList
    self.MELCORPlotFile = melcorCombinedInterface.MelcorApp.MelcorPlotFile


  def writeCsv(self,filen,workDir):
    """
      Output the parsed results into a CSV file
      @ In, filen, str, the file name of the CSV file
      @ In, workDir, str, current working directory
      @ Out, None
    """
    IOcsvfile=open(filen,'w+')
    fileDir = os.path.join(workDir,self.MELCORPlotFile)
    Time,Data,VarUdm = MCRBin(fileDir,self.VarSrch)
    dfTime = pd.DataFrame(Time, columns= ["Time"])
    dfData = pd.DataFrame(Data, columns = self.VarSrch)
    df = pd.concat([dfTime, dfData], axis=1, join='inner')
    df.to_csv(IOcsvfile,index=False, header=True)
