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
from CSVDiffer import CSVDiffer
from UnorderedCSVDiffer import UnorderedCSVDiffer
from XMLDiff import XMLDiff
from TextDiff import TextDiff
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

_missing_modules, _too_old_modules, _notQAModules = RavenUtils.checkForMissingModules()

class RavenFramework(Tester):

  @staticmethod
  def validParams():
    params = Tester.validParams()
    params.addRequiredParam('input',"The input file to use for this test.")
    params.addParam('output','',"List of output files that the input should create.")
    params.addParam('csv','',"List of csv files to check")
    params.addParam('UnorderedCsv','',"List of unordered csv files to check")
    params.addParam('xml','',"List of xml files to check")
    params.addParam('UnorderedXml','',"List of unordered xml files to check")
    params.addParam('xmlopts','',"Options for xml checking")
    params.addParam('text','',"List of generic text files to check")
    params.addParam('comment','-20021986',"Character or string denoting comments, all text to the right of the symbol will be ignored in the diff of text files")
    params.addParam('image','',"List of image files to check")
    params.addParam('rel_err','','Relative Error for csv files or floats in xml ones')
    params.addParam('required_executable','','Skip test if this executable is not found')
    params.addParam('required_libraries','','Skip test if any of these libraries are not found')
    params.addParam('minimum_library_versions','','Skip test if the library listed is below the supplied version (e.g. minimum_library_versions = \"name1 version1 name2 version2\")')
    params.addParam('skip_if_env','','Skip test if this environmental variable is defined')
    params.addParam('skip_if_OS','','Skip test if the operating system defined')
    params.addParam('test_interface_only',False,'Test the interface only (without running the driven code')
    params.addParam('check_absolute_value',False,'if true the values are compared in absolute value (abs(trueValue)-abs(testValue)')
    params.addParam('zero_threshold',sys.float_info.min*4.0,'it represents the value below which a float is considered zero (XML comparison only)')
    params.addParam('remove_whitespace',False,'Removes whitespace before comparing xml node text if True')
    params.addParam('expected_fail', False, 'if true, then the test should fails, and if it passes, it fails.')
    params.addParam('remove_unicode_identifier', False, 'if true, then remove u infront of a single quote')
    params.addParam('interactive', False, 'if true, then RAVEN will be run with interactivity enabled.')
    params.addParam('python3_only', False, 'if true, then only use with Python3')
    params.addParam('ignore_sign', False, 'if true, then only compare the absolute values')
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


  def __init__(self, name, params):
    Tester.__init__(self, name, params)
    self.check_files = self.specs['output'      ].split(" ") if len(self.specs['output'      ]) > 0 else []
    self.csv_files   = self.specs['csv'         ].split(" ") if len(self.specs['csv'         ]) > 0 else []
    self.xml_files   = self.specs['xml'         ].split(" ") if len(self.specs['xml'         ]) > 0 else []
    self.ucsv_files  = self.specs['UnorderedCsv'].split(" ") if len(self.specs['UnorderedCsv']) > 0 else []
    self.uxml_files  = self.specs['UnorderedXml'].split(" ") if len(self.specs['UnorderedXml']) > 0 else []
    self.text_files  = self.specs['text'        ].split(" ") if len(self.specs['text'        ]) > 0 else []
    self.img_files   = self.specs['image'       ].split(" ") if len(self.specs['image'       ]) > 0 else []
    self.all_files = self.check_files + self.csv_files + self.xml_files + self.ucsv_files + self.uxml_files + self.text_files + self.img_files
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
    runpath = self.getTestDir()
    return list(os.path.join(runpath,file) for file in self.all_files)

  def prepare(self, options = None):
    self.check_files = [os.path.join(self.specs['test_dir'],filename)  for filename in self.check_files]
    for filename in self.__getCreatedFiles():
      if os.path.exists(filename):
        os.remove(filename)

  def processResults(self, moose_dir,  options, output):
    expectedFail = self.specs['expected_fail']
    if not expectedFail:
      return self.rawProcessResults(moose_dir, options, output)
    else:
      output = self.rawProcessResults(moose_dir, options, output)
      if self.didPass():
        self.setStatus('Unexpected success',self.bucket_fail)
        return output
      else:
        self.setStatus(self.success_message, self.bucket_success)
        return output

  def rawProcessResults(self, moose_dir, options, output):
    missing = []
    for filename in self.check_files:
      if not os.path.exists(filename):
        missing.append(filename)

    if len(missing) > 0:
      self.setStatus('CWD '+os.getcwd()+' METHOD '+os.environ.get("METHOD","?")+' Expected files not created '+" ".join(missing),self.bucket_fail)
      return output

    #csv
    if len(self.specs["rel_err"]) > 0:
      csv_diff = CSVDiffer(self.specs['test_dir'],self.csv_files,relative_error=float(self.specs["rel_err"]))
    else:
      csv_diff = CSVDiffer(self.specs['test_dir'],self.csv_files)
    message = csv_diff.diff()
    if csv_diff.getNumErrors() > 0:
      self.setStatus(message,self.bucket_diff)
      return output

    #unordered csv
    checkAbsoluteValue = self.specs["check_absolute_value"]
    zeroThreshold = self.specs["zero_threshold"]
    if len(self.specs["rel_err"]) > 0:
      ucsv_diff = UnorderedCSVDiffer(self.specs['test_dir'],
                  self.ucsv_files,
                  relative_error = float(self.specs["rel_err"]),
                  absolute_check = checkAbsoluteValue,
                  zeroThreshold = zeroThreshold, ignore_sign=self.specs["ignore_sign"])
    else:
      ucsv_diff = UnorderedCSVDiffer(self.specs['test_dir'],
                  self.ucsv_files,
                  absolute_check = checkAbsoluteValue,
                  zeroThreshold = zeroThreshold, ignore_sign=self.specs["ignore_sign"])

    ucsv_same,ucsv_messages = ucsv_diff.diff()
    if not ucsv_same:
      self.setStatus(ucsv_messages, self.bucket_diff)
      return output

    #xml
    xmlopts = {}
    if len(self.specs["rel_err"]) > 0: xmlopts['rel_err'] = float(self.specs["rel_err"])
    xmlopts['zero_threshold'] = float(self.specs["zero_threshold"])
    xmlopts['unordered'     ] = False
    xmlopts['remove_whitespace'] = self.specs['remove_whitespace'] == True
    xmlopts['remove_unicode_identifier'] = self.specs['remove_unicode_identifier']
    if len(self.specs['xmlopts'])>0: xmlopts['xmlopts'] = self.specs['xmlopts'].split(' ')
    xml_diff = XMLDiff(self.specs['test_dir'],self.xml_files,**xmlopts)
    (xml_same,xml_messages) = xml_diff.diff()
    if not xml_same:
      self.setStatus(xml_messages, self.bucket_diff)
      return output

    #unordered xml
    xmlopts['unordered'] = True
    uxml_diff = XMLDiff(self.specs['test_dir'],self.uxml_files,**xmlopts)
    (uxml_same,uxml_messages) = uxml_diff.diff()
    if not uxml_same:
      self.setStatus(uxml_messages, self.bucket_diff)
      return output

    #text
    textOpts = {'comment': self.specs['comment']}
    textDiff = TextDiff(self.specs['test_dir'],self.text_files,**textOpts)
    (textSame,textMessages) = textDiff.diff()
    if not textSame:
      self.setStatus(textMessages, self.bucket_diff)
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

    self.setStatus(self.success_message, self.bucket_success)
    return output
