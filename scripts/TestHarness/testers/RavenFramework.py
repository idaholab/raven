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
from Tester import Tester
import OrderedCSVDiffer
import UnorderedCSVDiffer
import XMLDiff
import TextDiff
from RAVENImageDiff import ImageDiff
import RavenUtils
import os
import subprocess
import sys
import distutils.version
import platform

# Set this outside the class because the framework directory is constant for
#  each instance of this Tester, and in addition, there is a problem with the
#  path by the time you call it in __init__ that causes it to think its absolute
#  path is somewhere under tests/framework.
# Be aware that if this file changes its location, this variable should also be
#  changed.
myDir = os.path.dirname(os.path.realpath(__file__))
RAVEN_DIR = os.path.abspath(os.path.join(myDir, '..', '..', '..', 'framework'))

#Need to add the directory for AMSC for doing module checks.
os.environ["PYTHONPATH"] = os.path.join(RAVEN_DIR,'contrib') + os.pathsep + os.environ.get("PYTHONPATH","")


_missing_modules, _too_old_modules, _notQAModules = RavenUtils.checkForMissingModules()

class RavenFramework(Tester):

  @staticmethod
  def get_valid_params():
    params = Tester.get_valid_params()
    params.add_required_param('input',"The input file to use for this test.")
    params.add_param('output','',"List of output files that the input should create.")
    params.add_param('csv','',"List of csv files to check")
    params.add_param('UnorderedCsv','',"List of unordered csv files to check")
    params.add_param('xml','',"List of xml files to check")
    params.add_param('UnorderedXml','',"List of unordered xml files to check")
    params.add_param('xmlopts','',"Options for xml checking")
    params.add_param('text','',"List of generic text files to check")
    params.add_param('comment','-20021986',"Character or string denoting comments, all text to the right of the symbol will be ignored in the diff of text files")
    params.add_param('image','',"List of image files to check")
    params.add_param('rel_err','','Relative Error for csv files or floats in xml ones')
    params.add_param('required_executable','','Skip test if this executable is not found')
    params.add_param('required_libraries','','Skip test if any of these libraries are not found')
    params.add_param('minimum_library_versions','','Skip test if the library listed is below the supplied version (e.g. minimum_library_versions = \"name1 version1 name2 version2\")')
    params.add_param('skip_if_env','','Skip test if this environmental variable is defined')
    params.add_param('skip_if_OS','','Skip test if the operating system defined')
    params.add_param('test_interface_only',False,'Test the interface only (without running the driven code')
    params.add_param('check_absolute_value',False,'if true the values are compared to the tolerance directectly, instead of relatively.')
    params.add_param('zero_threshold',sys.float_info.min*4.0,'it represents the value below which a float is considered zero (XML comparison only)')
    params.add_param('remove_whitespace',False,'Removes whitespace before comparing xml node text if True')
    params.add_param('remove_unicode_identifier', False, 'if true, then remove u infront of a single quote')
    params.add_param('interactive', False, 'if true, then RAVEN will be run with interactivity enabled.')
    params.add_param('python3_only', False, 'if true, then only use with Python3')
    params.add_param('ignore_sign', False, 'if true, then only compare the absolute values')
    return params

  def getCommand(self, options):
    ravenflag = ''
    if self.specs['test_interface_only']:
      ravenflag += ' interfaceCheck '

    if self.specs['interactive']:
      ravenflag += ' interactiveCheck '

    if RavenUtils.inPython3():
      return "python3 " + self.driver + " " + ravenflag + self.specs["input"]
    else:
      return "python " + self.driver + " " + ravenflag + self.specs["input"]

  def __make_differ(self, spec_name, differ_class, extra=None):
    """
    This adds a differ if the spec_name has files.
    spec_name: string of the list of files to use with the differ.
    differ_class: subclass of Differ to use with the files.
    extra: dictionary of extra parameters
    """
    if len(self.specs[spec_name]) == 0:
      #No files, so quit
      return
    differ_params = dict(self.specs)
    differ_params["output"] = self.specs[spec_name]
    differ_params["type"] = differ_class.__name__
    if extra is not None:
      differ_params.update(extra)
    self.add_differ(differ_class(spec_name, differ_params,self.get_test_dir()))

  def __init__(self, name, params):
    Tester.__init__(self, name, params)
    self.check_files = self.specs['output'      ].split(" ") if len(self.specs['output'      ]) > 0 else []
    self.img_files   = self.specs['image'       ].split(" ") if len(self.specs['image'       ]) > 0 else []
    self.all_files = self.check_files + self.img_files
    self.__make_differ('csv', OrderedCSVDiffer.OrderedCSV)
    self.__make_differ('UnorderedCsv', UnorderedCSVDiffer.UnorderedCSV)
    self.__make_differ('xml', XMLDiff.XML, {"unordered":False})
    self.__make_differ('UnorderedXml', XMLDiff.XML, {"unordered":True})
    self.__make_differ('text', TextDiff.Text)
    self.required_executable = self.specs['required_executable']
    self.required_libraries = self.specs['required_libraries'].split(' ')  if len(self.specs['required_libraries']) > 0 else []
    self.minimum_libraries = self.specs['minimum_library_versions'].split(' ')  if len(self.specs['minimum_library_versions']) > 0 else []
    #for image tests, minimum library is always scipy 0.15.0
    if len(self.img_files)>0:
      self.minimum_libraries += ['scipy','0.15.0']
    self.required_executable = self.required_executable.replace("%METHOD%",os.environ.get("METHOD","opt"))
    self.specs['scale_refine'] = False
    self.driver = os.path.join(RAVEN_DIR,'Driver.py')

  def checkRunnable(self, option):
    missing = _missing_modules
    too_old = _too_old_modules
    # remove tests based on skipping criteria
    ## required module is missing
    if len(missing) > 0:
      self.setStatus('skipped (Missing python modules: '+" ".join(missing)+
                     " PYTHONPATH="+os.environ.get("PYTHONPATH","")+')',
                     self.bucket_skip)
      return False
    ## required module is present, but too old
    if len(too_old) > 0  and RavenUtils.checkVersions():
      self.setStatus('skipped (Old version python modules: '+" ".join(too_old)+
                     " PYTHONPATH="+os.environ.get("PYTHONPATH","")+')',
                     self.bucket_skip)
      return False
    ## an environment varible value causes a skip
    if len(self.specs['skip_if_env']) > 0:
      env_var = self.specs['skip_if_env']
      if env_var in os.environ:
        self.setStatus('skipped (found environmental variable "'+env_var+'")',
                       self.bucket_skip)
        return False
    ## OS
    if len(self.specs['skip_if_OS']) > 0:
      skip_os = [x.strip().lower() for x in self.specs['skip_if_OS'].split(',')]
      # get simple-name platform (options are Linux, Windows, Darwin, or SunOS that I've seen)
      currentOS = platform.system().lower()
      # replace Darwin with more expected "mac"
      if currentOS == 'darwin':
        currentOS = 'mac'
      if currentOS in skip_os:
        self.setStatus('skipped (OS is "{}")'.format(currentOS),
                       self.bucket_skip)
        return False
    for lib in self.required_libraries:
      found, message, version =  RavenUtils.moduleReport(lib,'')
      if not found:
        self.setStatus('skipped (Unable to import library: "'+lib+'")',
                       self.bucket_skip)
        return False
    if self.specs['python3_only'] and not RavenUtils.inPython3():
      self.setStatus('Python 3 only',
                     self.bucket_skip)
      return False

    i = 0
    if len(self.minimum_libraries) % 2:
      self.setStatus('skipped (libraries are not matched to versions numbers: '+str(self.minimum_libraries)+')',
                     self.bucket_skip)
      return False
    while i < len(self.minimum_libraries):
      libraryName = self.minimum_libraries[i]
      libraryVersion = self.minimum_libraries[i+1]
      found, message, actualVersion = RavenUtils.moduleReport(libraryName,libraryName+'.__version__')
      if not found:
        self.setStatus('skipped (Unable to import library: "'+libraryName+'")',
                       self.bucket_skip)
        return False
      if distutils.version.LooseVersion(actualVersion) < distutils.version.LooseVersion(libraryVersion):
        self.setStatus('skipped (Outdated library: "'+libraryName+'")',
                       self.bucket_skip)
        return False
      i+=2

    if len(self.required_executable) > 0 and \
       not os.path.exists(self.required_executable):
      self.setStatus('skipped (Missing executable: "'+self.required_executable+'")',
                     self.bucket_skip)
      return False
    try:
      if len(self.required_executable) > 0 and \
         subprocess.call([self.required_executable],stdout=subprocess.PIPE) != 0:
        self.setStatus('skipped (Failing executable: "'+self.required_executable+'")',
                      self.bucket_skip)
        return False
    except:
      self.setStatus('skipped (Error when trying executable: "'+self.required_executable+'")',
                     self.bucket_skip)
      return False
    filenameSet = set()
    duplicateFiles = []
    for filename in self.__getCreatedFiles():
      if filename not in filenameSet:
        filenameSet.add(filename)
      else:
        duplicateFiles.append(filename)
    if len(duplicateFiles) > 0:
      self.setStatus('[incorrect test] duplicated files specified: '+
                     " ".join(duplicateFiles),
                     self.bucket_skip)
      return False
    return True

  def __getCreatedFiles(self):
    """
      Returns all the files used by this test that need to be created
      by the test.  Note that they will be deleted at the start of running
      the test.
      @ Out, createdFiles, [str], list of files created by the test.
    """
    runpath = self.get_test_dir()
    remove_files = self.get_differ_remove_files(runpath)
    return remove_files+list(os.path.join(runpath,file) for file in self.all_files)

  def prepare(self, options = None):
    self.check_files = [os.path.join(self.specs['test_dir'],filename)  for filename in self.check_files]
    for filename in self.__getCreatedFiles():
      if os.path.exists(filename):
        os.remove(filename)

  def processResults(self, moose_dir, options, output):
    missing = []
    for filename in self.check_files:
      if not os.path.exists(filename):
        missing.append(filename)

    if len(missing) > 0:
      self.setStatus('CWD '+os.getcwd()+' METHOD '+os.environ.get("METHOD","?")+' Expected files not created '+" ".join(missing),self.bucket_fail)
      return output

    #image
    imageOpts = {}
    if 'rel_err'        in self.specs.keys(): imageOpts['rel_err'       ] = self.specs['rel_err'       ]
    if 'zero_threshold' in self.specs.keys(): imageOpts['zero_threshold'] = self.specs['zero_threshold']
    imgDiff = ImageDiff(self.specs['test_dir'],self.img_files,**imageOpts)
    (imgSame,imgMessages) = imgDiff.diff()
    if not imgSame:
      self.setStatus(imgMessages, self.bucket_diff)
      return output

    self.set_success()
    return output
