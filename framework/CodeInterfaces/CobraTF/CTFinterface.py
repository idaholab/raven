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


from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os
from ctfdata import ctfdata
from CodeInterfaceBaseClass import CodeInterfaceBase

class CobraTF(CodeInterfaceBase):
    def createNewInput(self, currentInputFiles, oriInputFiles, samplerType ,**Kwargs):
        # @ In, Kwargs
        # @ Out, current input file names & modified variables list
	from CTFparser import CTFparser

        # currently, CTF parsers only work with point sampler type
        if 'dynamicevent' in samplerType.lower():
        	raise IOError("Dynamic Even Tree-based sampling not implemented yet!")

        # 1. check the existence of current input files
        found = False
        for index, inputFile in enumerate(currentInputFiles):
		if inputFile.getExt() in self.getInputExtension():
 		  found = True
		  break
        if not found:
        	raise IOError('There is no CTF input file (*.inp)' )

        # 2. call input parser (get AbsFile???)
        ctf_parser = CTFparser(currentInputFiles[index].getAbsFile())  # import CTFparser

        # 3. print new input
        modifDict = Kwargs["SampledVars"]  # {LineNumPosition : value}
        ctf_parser.changeVariable(modifDict)  # {lineNumberPosition: value}
        modifiedDictionary = ctf_parser.Modif_Dict

        ctf_parser.printInput(currentInputFiles[index].getAbsFile())
        #ctf_parser.printInput(test1)

        return currentInputFiles

    def finalizeCodeOutput(self,command,output,workingDir):
      """
        This method is called by the RAVEN code at the end of each run (if the method is present, since it is optional).
        It can be used for those codes, that do not create CSV files to convert the whatever output format into a csv
        @ In, command, string, the command used to run the just ended job
        @ In, output, string, the Output name root
        @ In, workingDir, string, current working dir
        @ Out, output, string, optional, present in case the root of the output file gets changed in this method.
      """
      outfile  = os.path.join(workingDir,output+'.out')
      outputobj= ctfdata(outfile)
      # convert to csv file
      outputobj.writeCSV(os.path.join(workingDir,output+'.csv'))


    def generateCommand(self,inputFiles,executable,clargs=None,fargs=None):
     """
      This method is used to retrieve the command (in tuple format) needed to launch the Code.
      See base class.  Collects all the clargs and the executable to produce the command-line call.
      Returns tuple of commands and base file name for run.
      Commands are a list of tuples, indicating parallel/serial and the execution command to use.
      @ In, inputFiles, list, List of input files (length of the list depends on the number of inputs have been added in the Step is running this code)
      @ In, executable, string, executable name with absolute path (e.g. /home/path_to_executable/code.exe)
      @ In, clargs, dict, optional, dictionary containing the command-line flags the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 i0extension =0 .inp0/ >< /Code >)
      @ In, fargs, dict, optional, a dictionary containing the auxiliary input file variables the user can specify in the input (e.g. under the node < Code >< clargstype =0 input0arg =0 aux0extension =0 .aux0/ >< /Code >)
      @ Out, returnCommand, tuple, tuple containing the generated command. returnCommand[0] is the command to run the code (string), returnCommand[1] is the name of the output root
     """
     found = False
     for index, inputFile in enumerate(inputFiles):
       if inputFile.getExt() in self.getInputExtension():
         found = True
         break
     if not found:
       raise IOError('There is no CTF input file (*.inp)' )

     #commandToRun = executable + ' -i ' + inputFiles[index].getFilename() + ' -o ' + outputfile  + '.o' + ' -r ' + outputfile  + '.r' + addflags
     commandToRun = executable + ' ' + inputFiles[index].getFilename()
     commandToRun = commandToRun.replace("\n"," ")
     #commandToRun  = re.sub("\s\s+" , " ", commandToRun )
     returnCommand = [('parallel', commandToRun)]
     output = inputFiles[index].getBase() + '.ctf'

     return returnCommand,output

