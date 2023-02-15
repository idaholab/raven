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
RavenFramework is a tool to test raven inputs.
"""
from __future__ import absolute_import
import os
import subprocess
import sys
import platform
from Tester import Tester
import OrderedCSVDiffer
import UnorderedCSVDiffer
import XMLDiff
import TextDiff
import ExistsDiff
import RAVENImageDiff

# Set this outside the class because the framework directory is constant for
#  each instance of this Tester, and in addition, there is a problem with the
#  path by the time you call it in __init__ that causes it to think its absolute
#  path is somewhere under tests/framework.
# Be aware that if this file changes its location, this variable should also be
#  changed.
myDir = os.path.dirname(os.path.realpath(__file__))
RAVENDIR = os.path.abspath(os.path.join(myDir, '..', '..', '..', 'ravenframework'))
RAVENROOTDIR = os.path.abspath(os.path.join(myDir, '..', '..', '..'))

#add path so framework is found.
sys.path.append(os.path.abspath(os.path.dirname(RAVENDIR)))

#Need to add the directory for AMSC for doing module checks.
os.environ["PYTHONPATH"] = os.path.join(RAVENDIR, '..', 'install') +\
  os.pathsep + os.environ.get("PYTHONPATH", "")

scriptDir = os.path.abspath(os.path.join(RAVENDIR, '..', 'scripts'))
sys.path.append(scriptDir)
import library_handler
sys.path.pop()

_missingModules, _notQAModules = library_handler.checkLibraries()
_checkVersions = library_handler.checkVersions()

class RavenFramework(Tester):
  """
  RavenFramework is the class to use for testing standard raven inputs.
  """

  @staticmethod
  def get_valid_params():
    """
      Returns the parameters that can be used for this class.
      @ In, None
      @ Out, params, _ValidParameters, return the parameters.
    """
    params = Tester.get_valid_params()
    params.add_required_param('input', "The input file to use for this test.")
    params.add_param('output', '', "List of output files that the input should create.")
    params.add_param('csv', '', "List of csv files to check")
    params.add_param('UnorderedCsv', '', "List of unordered csv files to check")
    params.add_param('xml', '', "List of xml files to check")
    params.add_param('UnorderedXml', '', "List of unordered xml files to check")
    params.add_param('xmlopts', '', "Options for xml checking")
    params.add_param('text', '', "List of generic text files to check")
    params.add_param('comment', '-20021986', "Character or string denoting "+
                     "comments, all text to the right of the symbol will be "+
                     "ignored in the diff of text files")
    params.add_param('image', '', "List of image files to check")
    params.add_param('rel_err', '', 'Relative Error for csv files or floats in xml ones')
    params.add_param('required_executable', '', 'Skip test if this executable is not found')
    params.add_param('required_libraries', '', 'Skip test if any of these libraries are not found')
    params.add_param('minimum_library_versions', '',
                     'Skip test if the library listed is below the supplied'+
                     ' version (e.g. minimum_library_versions = \"name1 version1 name2 version2\")')
    params.add_param('skip_if_env', '', 'Skip test if this environmental variable is defined')
    params.add_param('test_interface_only', False,
                     'Test the interface only (without running the driven code')
    params.add_param('check_absolute_value', False,
                     'if true the values are compared to the tolerance '+
                     'directectly, instead of relatively.')
    params.add_param('zero_threshold', sys.float_info.min*4.0,
                     'it represents the value below which a float is'+
                     'considered zero (XML comparison only)')
    params.add_param('remove_whitespace', False,
                     'Removes whitespace before comparing xml node text if True')
    params.add_param('remove_unicode_identifier', False,
                     'if true, then remove u infront of a single quote')
    params.add_param('interactive', False,
                     'if true, then RAVEN will be run with interactivity enabled.')
    params.add_param('python3_only', False, 'if true, then only use with Python3')
    params.add_param('ignore_sign', False, 'if true, then only compare the absolute values')
    return params

  def get_command(self):
    """
      Gets the raven command to run this test.
      @ In, None
      @ Out, get_command, string, command to run.
    """
    ravenflag = ''
    if self.specs['test_interface_only']:
      ravenflag += ' interfaceCheck '

    if self.specs['interactive']:
      ravenflag += ' interactiveCheck '

    return self._get_python_command() + " " + self.driver + " " + ravenflag + self.specs["input"]

  def __make_differ(self, specName, differClass, extra=None):
    """
      This adds a differ if the specName has files.
      @ In, specName, string of the list of files to use with the differ.
      @ In, differClass, subclass of Differ, for use with the files.
      @ In, extra, dictionary, extra parameters
      @ Out, None
    """
    if len(self.specs[specName]) == 0:
      #No files, so quit
      return
    differParams = dict(self.specs)
    differParams["output"] = self.specs[specName]
    differParams["type"] = differClass.__name__
    if extra is not None:
      differParams.update(extra)
    self.add_differ(differClass(specName, differParams, self.get_test_dir()))

  def __init__(self, name, params):
    Tester.__init__(self, name, params)
    self.all_files = []
    self.__make_differ('output', ExistsDiff.Exists)
    self.__make_differ('csv', OrderedCSVDiffer.OrderedCSV)
    self.__make_differ('UnorderedCsv', UnorderedCSVDiffer.UnorderedCSV)
    self.__make_differ('xml', XMLDiff.XML, {"unordered":False})
    self.__make_differ('UnorderedXml', XMLDiff.XML, {"unordered":True})
    self.__make_differ('text', TextDiff.Text)
    self.__make_differ('image', RAVENImageDiff.ImageDiffer)
    self.required_executable = self.specs['required_executable']
    self.required_libraries = self.specs['required_libraries'].split(' ')\
      if len(self.specs['required_libraries']) > 0 else []
    self.minimum_libraries = self.specs['minimum_library_versions'].split(' ')\
      if len(self.specs['minimum_library_versions']) > 0 else []
    self.required_executable = self.required_executable.replace("%METHOD%",
                                                                os.environ.get("METHOD", "opt"))
    self.specs['scale_refine'] = False
    self.driver = os.path.join(RAVENROOTDIR, 'raven_framework.py')

  def check_runnable(self):
    """
      Checks if this test can run.
      @ In, None
      @ Out, check_runnable, boolean, if True can run this test.
    """
    # remove tests based on skipping criteria
    ## required module is missing
    if _missingModules:
      self.set_fail('skipped (Missing python modules: '+" ".join([m[0] for m in _missingModules])+
                    " PYTHONPATH="+os.environ.get("PYTHONPATH", "")+')')
      return False
    ## required module is present, but too old
    if _notQAModules and _checkVersions:
      self.set_fail('skipped (Incorrectly versioned python modules: ' +
                    " ".join(['required {}-{}, but found {}'.format(*m) for m in _notQAModules]) +
                    " PYTHONPATH="+os.environ.get("PYTHONPATH", "")+')')
      return False
    ## an environment varible value causes a skip
    if len(self.specs['skip_if_env']) > 0:
      envVar = self.specs['skip_if_env']
      if envVar in os.environ:
        self.set_skip('skipped (found environmental variable "'+envVar+'")')
        return False
    for lib in self.required_libraries:
      found, _, _ = library_handler.checkSingleLibrary(lib)
      if not found:
        self.set_skip('skipped (Unable to import library: "{}")'.format(lib))
        return False
    if self.specs['python3_only'] and not library_handler.inPython3():
      self.set_skip('Python 3 only')
      return False

    i = 0
    if len(self.minimum_libraries) % 2:
      self.set_skip('skipped (libraries are not matched to versions numbers: '
                    +str(self.minimum_libraries)+')')
      return False
    while i < len(self.minimum_libraries):
      libraryName = self.minimum_libraries[i]
      libraryVersion = self.minimum_libraries[i+1]
      found, _, actualVersion = library_handler.checkSingleLibrary(libraryName, version='check')
      if not found:
        self.set_skip('skipped (Unable to import library: "'+libraryName+'")')
        return False
      if library_handler.parseVersion(actualVersion) < \
         library_handler.parseVersion(libraryVersion):
        self.set_skip('skipped (Outdated library: "'+libraryName+'" needed version '+str(libraryVersion)+' but had version '+str(actualVersion)+')')
        return False
      i += 2

    if len(self.required_executable) > 0 and \
       not os.path.exists(self.required_executable):
      self.set_skip('skipped (Missing executable: "'+self.required_executable+'")')
      return False
    try:
      if len(self.required_executable) > 0 and \
         subprocess.call([self.required_executable], stdout=subprocess.PIPE) != 0:
        self.set_skip('skipped (Failing executable: "'+self.required_executable+'")')
        return False
    except Exception as exp:
      self.set_skip('skipped (Error when trying executable: "'
                    +self.required_executable+'")'+str(exp))
      return False
    filenameSet = set()
    duplicateFiles = []
    for filename in self.__get_created_files():
      if filename not in filenameSet:
        filenameSet.add(filename)
      else:
        duplicateFiles.append(filename)
    if len(duplicateFiles) > 0:
      self.set_skip('[incorrect test] duplicated files specified: '+
                    " ".join(duplicateFiles))
      return False
    return True

  def __get_created_files(self):
    """
      Returns all the files used by this test that need to be created
      by the test.  Note that they will be deleted at the start of running
      the test.
      @ In, None
      @ Out, createdFiles, [str], list of files created by the test.
    """
    runpath = self.get_test_dir()
    removeFiles = self.get_differ_remove_files()
    return removeFiles+list(os.path.join(runpath, file) for file in self.all_files)

  def prepare(self):
    """
      Get the test ready to run by removing files that should be created.
      @ In, None
      @ Out, None
    """
    for filename in self.__get_created_files():
      if os.path.exists(filename):
        os.remove(filename)

  def process_results(self, _):
    """
      Check to see if the test has passed.
      @ In, ignored, string, output of test.
      @ Out, None
    """
    self.set_success()
