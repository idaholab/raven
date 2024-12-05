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
Created Feb 9th, 2024

@author: alfoa
"""

if __name__ == '__main__':
  import argparse
  import re
  from ravenframework.CodeInterfaceClasses.SERPENT import serpentOutputParser as op
  # read and process input arguments
  # ================================
  inpPar = argparse.ArgumentParser(description = 'Utility to generate RAVEN variables from Serpent output files')
  inpPar.add_argument('-fileTypes', nargs=1, required=True, help='Comma-separated list of output file '
                                                                 'types from which variable names need to be extracted. Currently types available are:'
                                                                 '"ResultsReader","DepletionReader","DetectorReader","DepmtxReader".', )
  inpPar.add_argument('-fileRoot', nargs=1, required=True, help='File name root from which all the output files are going to be inferred.', )
  inpPar.add_argument('-o', nargs=1, required=False, help='Output file name')
  inp = inpPar.parse_args()

  outputXmlName = inp.o[0] if inp.o is not None else "ravenOutputVariables.xml"

  # get output files types
  outputFilesTypes = [e.strip() for e in inp.fileTypes[0].split(",")]
  outputFileRoot = inp.fileRoot[0]
  if 'ResultsReader' not in outputFilesTypes:
    raise IOError(f' ERROR: <Serpent File Type> ResultsReader must be present!')
  for ft in outputFilesTypes:
    if ft not in op.serpentOutputAvailableTypes:
      raise IOError(f' ERROR: <Serpent File Type> {ft} not supported! Available types are {", ".join(op.serpentOutputAvailableTypes)}!!')

  parser = op.SerpentOutputParser(outputFilesTypes, outputFileRoot)
  # get results
  print("Reading variables from '_res.m' file!")
  results, _ = parser._resultsReader()
  # get variables from result file
  resultFileVariables = list(results.keys())
  detectorFileVariables = None
  dplFileVariables = None
  depmtxFileVariables = None
  for ft in outputFilesTypes:
    if ft == 'DetectorReader':
      print("Reading variables from '_det0.m' file!")
      detectorFileVariables = list(parser._detectorReader(1).keys())
    elif ft == 'DepletionReader':
      print("Reading variables from '_dep.m' file!")
      dplFileVariables = list(parser._depletionReader().keys())
    elif ft == 'DepmtxReader':
      print("Reading variables from '_depmtx_*.m' files!")
      depmtxFileVariables = list(parser._depmtxReader(1).keys())


  with open(outputXmlName,"w") as fo:
    fo.write(" <VariableGroups>\n")
    resultFileVariablesGroup = '   <Group name="resVariables">\n  ' + ",\n  ".join(resultFileVariables) + '\n   </Group>\n'
    fo.write(resultFileVariablesGroup)
    if detectorFileVariables is not None:
      detectorFileVariablesGroup = '   <Group name="detectorsVariables">\n  ' + ",\n  ".join(detectorFileVariables) + '\n   </Group>\n'
      fo.write(detectorFileVariablesGroup)
    if dplFileVariables is not None:
      dplFileVariablesGroup = '   <Group name="depletionVariables">\n  ' + ",\n  ".join(dplFileVariables) + '\n   </Group>\n'
      fo.write(dplFileVariablesGroup)
    if depmtxFileVariables is not None:
      depmtxFileVariablesGroup = '   <Group name="depmtxVariables">\n  ' + ",\n  ".join(depmtxFileVariables) + '\n   </Group>\n'
      fo.write(depmtxFileVariablesGroup)

    fo.write(" </VariableGroups>")
