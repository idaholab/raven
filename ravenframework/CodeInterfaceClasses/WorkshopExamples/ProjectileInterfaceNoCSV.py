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
Created on Oct 31, 2020

@author: Andrea Alfonsi

comments: Interface for Projectile Code without the creation of
          a CSV but with the direct transfer of data to RAVEN in
          finalizeCodeOutput method
"""
import os
import numpy

from ravenframework.CodeInterfaceBaseClass import CodeInterfaceBase
from .ProjectileInterface import Projectile

class ProjectileNoCSV(Projectile):
  """
    Provides code to interface RAVEN to Projectile without the need of a CSV
    The only method that changes is the finalizeCodeOutput (we return the data directly)
  """
  def finalizeCodeOutput(self, command, output, workingDir):
    """
      Called by RAVEN to modify output files (if needed) so that they are in a proper form.
      In this case, the data are directly returned as a dictionary
      @ In, command, string, the command used to run the just ended job
      @ In, output, string, the Output name root
      @ In, workingDir, string, current working dir
      @ Out, output, dict, the dictionary containing the data
    """
    # open output file
    outfileName = os.path.join(workingDir,output+".txt" )
    headers, data = self._readOutputData(outfileName)
    dat = numpy.asarray(data).T
    output = {var:dat[i,:] for (i, var) in enumerate(headers)}
    return output
