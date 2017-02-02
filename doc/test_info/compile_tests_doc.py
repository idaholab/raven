import os,sys
import xml.etree.ElementTree as ET



class Test(object):
  def __init__(self,absFile,name,author,classesTested,desc,created,reqs=None,analytic=None):
    #TODO split "name" into path and name
    path,fname          = os.path.split(absFile)
    self.path           = path
    self.filename       = fname
    self.name           = name
    self.author         = author
    self.classesTested  = classesTested
    self.description    = desc
    self.created        = created
    self.requirements   = reqs
    self.analytic       = analytic

  def __setattr__(self,name,value):
    if type(value)==str:
      value = value.replace('_','\_').replace('#','\#').replace('%','\%')
    super(Test,self).__setattr__(name,value)

def parseFile(path,filename):
  absFile = os.path.join(path,filename)
  root = ET.parse(absFile).getroot()
  testInfoNode = root.find('TestInfo')
  if testInfoNode is None:
    print 'WARNING: test has no TestInfo block:',absFile
    test = Test('None/None','None','None','None','This test has no TestInfo block (yet).','None')
    return None
  else:
    name = testInfoNode.find('name').text
    author = testInfoNode.find('author').text
    classesTested = testInfoNode.find('classesTested').text
    desc = testInfoNode.find('description').text
    created = testInfoNode.find('created').text
    reqs = testInfoNode.find('requirements').text if testInfoNode.find('requirements') is not None else None
    analytic = testInfoNode.find('analytic').text if testInfoNode.find('analytic') is not None else None
    test = Test(absFile,name,author,classesTested,desc,created,reqs,analytic)
    #TODO revisions
  return test

#
# \subsection{path}
# \subsubsection{name}
#   \begin{itemize}
#     \item filename:
#     \item requirements:
#     \item classes tested:
#     \item created:
#     \item author:
#     \item description:
#     \item analytic:
#   \end{itemize}
def writeTexEntry(path,test):
  msg  = '    \\subsubsection{'+test.name+'}\n'
  msg += '      '+test.description+'\n'
  if test.analytic is not None:
    msg += '\n      '+test.analytic+'\n'
  msg += '      \\begin{itemize}\n'
  msg += '          \\item filename: '+test.filename+'\n'
  if test.requirements is not None:
    msg += '          \\item requirements: '+test.filename+'\n'
  msg += '          \\item classes tested: '+test.classesTested+'\n' #TODO list formatting
  msg += '          \\item created: '+test.created+'\n' #TODO date formatting
  msg += '          \\item author: '+test.author+'\n'
  msg += '      \\end{itemize}\n'
  return msg


if __name__=='__main__':
  cwd = os.getcwd()
  devDir = os.path.join(cwd,'..','..','developer_tools')
  sys.path.append(devDir)
  import get_coverage_tests as gct
  tests = gct.getRegressionTests()
  outfile = file('test_doc_entries.tex','w')
  for folder,files in tests.items():
    first = True
    for f in files:
      test = parseFile(folder,f)
      if test is None:
        continue
      if first:
        outfile.writelines('  \\subsection{'+test.path+'}\n')
        first = False
      tex = writeTexEntry(folder,test)
      outfile.writelines(tex)

