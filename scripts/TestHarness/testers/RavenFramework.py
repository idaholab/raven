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
import os
import subprocess
import sys
import distutils.version
import platform
from Tester import Tester
import OrderedCSVDiffer
import UnorderedCSVDiffer
import XMLDiff
import TextDiff
import ExistsDiff
from RAVENImageDiff import ImageDiff
import RavenUtils

# Set this outside the class because the framework directory is constant for
#  each instance of this Tester, and in addition, there is a problem with the
#  path by the time you call it in __init__ that causes it to think its absolute
#  path is somewhere under tests/framework.
# Be aware that if this file changes its location, this variable should also be
#  changed.
myDir = os.path.dirname(os.path.realpath(__file__))
RAVEN_DIR = os.path.abspath(os.path.join(myDir, '..', '..', '..', 'framework'))

#Need to add the directory for AMSC for doing module checks.
os.environ["PYTHONPATH"] = os.path.join(RAVEN_DIR, 'contrib') +\
  os.pathsep + os.environ.get("PYTHONPATH", "")


_missing_modules, _too_old_modules, _notQAModules = RavenUtils.check_for_missing_modules()

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
    params.add_param('skip_if_OS', '', 'Skip test if the operating system defined')
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

  def __make_differ(self, spec_name, differ_class, extra=None):
    """
      This adds a differ if the spec_name has files.
      @ In, spec_name, string of the list of files to use with the differ.
      @ In, differ_class, subclass of Differ, for use with the files.
      @ In, extra, dictionary, extra parameters
      @ Out, None
    """
    if len(self.specs[spec_name]) == 0:
      #No files, so quit
      return
    differ_params = dict(self.specs)
    differ_params["output"] = self.specs[spec_name]
    differ_params["type"] = differ_class.__name__
    if extra is not None:
      differ_params.update(extra)
    self.add_differ(differ_class(spec_name, differ_params, self.get_test_dir()))

  def __init__(self, name, params):
    Tester.__init__(self, name, params)
    self.img_files = self.specs['image'].split(" ") if len(self.specs['image']) > 0 else []
    self.all_files = self.img_files
    self.__make_differ('output', ExistsDiff.Exists)
    self.__make_differ('csv', OrderedCSVDiffer.OrderedCSV)
    self.__make_differ('UnorderedCsv', UnorderedCSVDiffer.UnorderedCSV)
    self.__make_differ('xml', XMLDiff.XML, {"unordered":False})
    self.__make_differ('UnorderedXml', XMLDiff.XML, {"unordered":True})
    self.__make_differ('text', TextDiff.Text)
    self.required_executable = self.specs['required_executable']
    self.required_libraries = self.specs['required_libraries'].split(' ')\
      if len(self.specs['required_libraries']) > 0 else []
    self.minimum_libraries = self.specs['minimum_library_versions'].split(' ')\
      if len(self.specs['minimum_library_versions']) > 0 else []
    #for image tests, minimum library is always scipy 0.15.0
    if len(self.img_files) > 0:
      self.minimum_libraries += ['scipy', '0.15.0']
    self.required_executable = self.required_executable.replace("%METHOD%",
                                                                os.environ.get("METHOD", "opt"))
    self.specs['scale_refine'] = False
    self.driver = os.path.join(RAVEN_DIR, 'Driver.py')

  def check_runnable(self):
    """
      Checks if this test can run.
      @ In, None
      @ Out, check_runnable, boolean, if True can run this test.
    """
    missing = _missing_modules
    too_old = _too_old_modules
    # remove tests based on skipping criteria
    ## required module is missing
    if len(missing) > 0:
      self.set_fail('skipped (Missing python modules: '+" ".join(missing)+
                    " PYTHONPATH="+os.environ.get("PYTHONPATH", "")+')')
      return False
    ## required module is present, but too old
    if len(too_old) > 0  and RavenUtils.check_versions():
      self.set_fail('skipped (Old version python modules: '+" ".join(too_old)+
                    " PYTHONPATH="+os.environ.get("PYTHONPATH", "")+')')
      return False
    ## an environment varible value causes a skip
    if len(self.specs['skip_if_env']) > 0:
      env_var = self.specs['skip_if_env']
      if env_var in os.environ:
        self.set_skip('skipped (found environmental variable "'+env_var+'")')
        return False
    ## OS
    if len(self.specs['skip_if_OS']) > 0:
      skip_os = [x.strip().lower() for x in self.specs['skip_if_OS'].split(',')]
      # get simple-name platform (options are Linux, Windows, Darwin, or SunOS that I've seen)
      current_os = platform.system().lower()
      # replace Darwin with more expected "mac"
      if current_os == 'darwin':
        current_os = 'mac'
      if current_os in skip_os:
        self.set_skip('skipped (OS is "{}")'.format(current_os))
        return False
    for lib in self.required_libraries:
      found, _, _ = RavenUtils.module_report(lib, '')
      if not found:
        self.set_skip('skipped (Unable to import library: "'+lib+'")')
        return False
    if self.specs['python3_only'] and not RavenUtils.in_python_3():
      self.set_skip('Python 3 only')
      return False

    i = 0
    if len(self.minimum_libraries) % 2:
      self.set_skip('skipped (libraries are not matched to versions numbers: '
                    +str(self.minimum_libraries)+')')
      return False
    while i < len(self.minimum_libraries):
      library_name = self.minimum_libraries[i]
      library_version = self.minimum_libraries[i+1]
      found, _, actual_version = RavenUtils.module_report(library_name, library_name+'.__version__')
      if not found:
        self.set_skip('skipped (Unable to import library: "'+library_name+'")')
        return False
      if distutils.version.LooseVersion(actual_version) < \
         distutils.version.LooseVersion(library_version):
        self.set_skip('skipped (Outdated library: "'+library_name+'")')
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
    filename_set = set()
    duplicate_files = []
    for filename in self.__get_created_files():
      if filename not in filename_set:
        filename_set.add(filename)
      else:
        duplicate_files.append(filename)
    if len(duplicate_files) > 0:
      self.set_skip('[incorrect test] duplicated files specified: '+
                    " ".join(duplicate_files))
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
    remove_files = self.get_differ_remove_files()
    return remove_files+list(os.path.join(runpath, file) for file in self.all_files)

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

    #image
    image_opts = {}
    if 'rel_err'        in self.specs.keys():
      image_opts['rel_err'] = self.specs['rel_err']
    if 'zero_threshold' in self.specs.keys():
      image_opts['zero_threshold'] = self.specs['zero_threshold']
    img_diff = ImageDiff(self.specs['test_dir'], self.img_files, **image_opts)
    (img_same, img_messages) = img_diff.diff()
    if not img_same:
      self.set_diff(img_messages)
      return

    self.set_success()
