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
import os,sys,subprocess
import tempfile
import get_coverage_tests

scriptDir = os.path.dirname(os.path.abspath(__file__))
conversionDir = os.path.join(scriptDir,'..','scripts','conversionScripts')
termproc = subprocess.Popen('tput cols',shell=True,stdout=subprocess.PIPE)
tlen = int(termproc.communicate()[0])
maxlen = min(100,tlen)

def validateTests():
  """
    Runs validation tests on regression tests and displays results.
    @ In, None
    @ Out, int, number of failed tests.
  """
  print 'Beginning test validation...'
  tests = get_coverage_tests.getRegressionTests(skipExpectedFails=True)
  res=[0,0,0] #run, pass, fail
  failed={}
  devnull = open(os.devnull, "wb")
  for dir,files in tests.items():
    #print directory being checked'
    checkmsg = ' Directory: '+dir
    print colors.neutral+checkmsg.rjust(maxlen,'-')
    #check files in directory
    for f in files:
      fullpath = os.path.join(dir,f)
      res[0]+=1
      startmsg =  'Validating '+f
      #expand external XML nodes
      # - first, though, check if the backup file already exists
      if os.path.isfile(fullpath+'.bak'):
        print colors.neutral+'Could not check for ExternalXML since a backup file exists! Please remove it to validate.'
      # Since directing output of shell commands to file /dev/null is problematic on Windows, this equivalent form is used
      #   which is better supported on all platforms.
      cmd = 'python '+os.path.join(conversionDir,'externalXMLNode.py')+' '+fullpath
      result = subprocess.call(cmd, shell = True, stdout=devnull)
      err = 'Error running ' + cmd

      print colors.neutral+startmsg,
      if result == 0:
        #run xmllint
        cmd = 'xmllint --noout --schema '+os.path.join(scriptDir,'XSDSchemas','raven.xsd')+' '+fullpath
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #collect output
        out,err = proc.communicate()
        result = proc.returncode
      if result == 0: #success
        res[1]+=1
        endmsg = 'validated'
        endcolor = colors.ok
        postprint=''
      else:
        res[2]+=1
        endmsg = 'FAILED'
        endcolor = colors.fail
        postprint= colors.fail+err
        if dir not in failed.keys(): failed[dir]=[]
        failed[dir].append(f)
      #print 'maxlen: %i len(start): %i len(end): %i' %(maxlen,len(startmsg),len(endmsg))
      print colors.neutral+''.rjust(maxlen-len(startmsg)-len(endmsg)-1,'.')+endcolor+endmsg+colors.neutral
      if len(postprint)>0: print postprint + colors.neutral
      #return externalNode xmls
      os.system('mv '+fullpath+'.bak '+fullpath)
  print colors.neutral+''
  print '-----------------------------------------------------------------------'
  print '                           Failure Summary'
  print '-----------------------------------------------------------------------'
  for dir,files in failed.items():
    for f in files:
      print colors.fail+os.path.join(dir,f)
  print colors.neutral+''
  print 'Validated: '+colors.ok+str(res[1])+colors.neutral+' Failed: '+colors.fail+str(res[2])+colors.neutral+' Run:',res[0]
  return res[2]

class colors:
  ok       = '\033[92m'
  fail     = '\033[91m'
  neutral  = '\033[0m'

if __name__=='__main__':
  sys.exit(validateTests())
