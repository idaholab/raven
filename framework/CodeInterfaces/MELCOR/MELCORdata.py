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
  @author: Matteo D'Onorio (University of Rome La Sapienza),
           Fabio Gianneti (University of Rome La Sapienza),
           Andrea Alfonsi (INL)

  Modified on January 24, 2018
  @author: Violet Olson
           Thomas Riley (Oregon State University)
           Robert Shannon (Oregon State University)
  Change Summary: Added Control Function parsing

  Modified on July 4, 2018
  @author: Matteo D'Onorio (University of Rome La Sapienza)
  Change Summary: Added Function to write CSV file from PTF
                  All the functions for the parsing have been removed
  
"""
from __future__ import division, print_function, unicode_literals, absolute_import
from melcor_tools import MCR_bin
import pandas as pd
import warnings
warnings.simplefilter('default',DeprecationWarning)
import re
import copy

class MELCORdata:
  """
    class that parses output of MELCOR 2.1 output file and reads in trip, minor block and write a csv file
    For now, Only the data associated to control volumes and control functions are parsed and output
  """
  def __init__(self,filen):
    """
      Constructor
      @ In, filen, FileObject, the file to parse
      @ Out, None
    """
    self.lines      = open(filen,"r").readlines()
    timeBlocks      = self.getTimeBlocks()
    self.timeParams = {}
    volForEachTime  = self.returnVolumeHybro(timeBlocks)
    self.timeParams.update(volForEachTime)
    self.functions  = self.returnControlFunctions(timeBlocks)

  def writeCsv(self,filen,filen2):
    """
      Output the parsed results into a CSV file
      @ In, filen, str, the file name of the CSV file
      @ Out, None
    """
    IOcsvfile=open(filen,'w+')
    file_dir = filen2
    Var_srch=['CVH-P_1','CVH-P_132','CVH-P_136','CVH-P_137','CVH-P_138','CVH-PPART.3_132','CVH-PPART.4_132','CVH-PPART.5_132','CVH-PPART.6_132','CVH-PPART.7_132','CVH-PPART.8_132','CVH-PPART.9_132','CVH-TLIQ_136','CVH-TVAP_136','CVH-TVAP_137', \
 'CFVALU_25','COR-TPN_1','COR-TPN_2','COR-TPN_3','COR-TPN_4','COR-TPN_5','COR-TCL_115','COR-TCL_116','COR-TCL_117','COR-TCL_118','COR-TCL_119','COR-TCL_120','COR-TCL_121','COR-TCL_122','COR-TCL_123','COR-TFU_115','COR-TFU_116','COR-TFU_117', \
 'COR-TFU_118','COR-TFU_119','COR-TFU_120','COR-TFU_121','COR-TFU_122', 'COR-DMH2-B4C','COR-DMH2-SS','COR-DMH2-TOT','COR-DMH2-ZIRC','COR-MCRP-TOT','COR-MSS-TOT','COR-MSSOX-TOT','COR-MUO2-TOT','COR-MZR-TOT','COR-MZRO2-TOT'] # !!!! Each element must be unique

    Time,Data,Var_Udm = MCR_bin(file_dir,Var_srch)
    df_time = pd.DataFrame(Time, columns= ["Time"])
    df_data = pd.DataFrame(Data, columns = Var_srch)
    df = pd.concat([df_time, df_data], axis=1, join='inner')
    df.to_csv(IOcsvfile,index=False, header=True)
