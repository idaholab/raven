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
    @ Out, None
  """
  print 'Beginning test validation...'
  tests = get_coverage_tests.getRegressionTests(skipThese=['test_rom_trainer.xml'],skipExpectedFails=True)
  res=[0,0,0] #run, pass, fail
  failed={}
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
      os.system('python '+os.path.join(conversionDir,'externalXMLNode.py')+' '+fullpath + '> /dev/null')
      #run xmllint
      print colors.neutral+startmsg,
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

class colors:
  ok       = '\033[92m'
  fail     = '\033[91m'
  neutral  = '\033[0m'

if __name__=='__main__':
  validateTests()
