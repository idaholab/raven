import os, sys

#find TreeStructure module
utilsDir = os.path.normpath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir,'framework','utils'))
sys.path.append(utilsDir)
import TreeStructure as TS

def convert(fileName):
  """
    Converts input files from standard RAVEN XML to xml-converted GetPot.  Produces a ".i" file in the end.
    @ In, fileName, the name for the XML to convert
    @ Out, None
  """
  tree = TS.parse(fileName)
  return tree.printGetPot()


if __name__=='__main__':
  if len(sys.argv) != 2:
      raise IOError('Expected one argument (the file to convert) but instead got %i: %s' %(len(sys.argv)-1,sys.argv[1:]))
  fName = sys.argv[1]
  if not os.path.isfile(fName):
    raise IOError('ERROR: File not found:',fName)
  toPrint = convert(fName)
  getpotName = fName.split('.')[0]+'.i'
  file(getpotName,'w').writelines(toPrint)
  print 'GetPot conversion written to',os.path.normpath(os.path.join(os.getcwd(),getpotName))

