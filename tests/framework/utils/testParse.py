import os,sys
import xml.etree.ElementTree as ET

utilsDir = os.path.normpath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir,os.pardir,'framework','utils'))
sys.path.append(utilsDir)

import TreeStructure as TS

results = {'failed':0,'passed':0}

def checkSameFile(a,b):
  genA = iter(a)
  genB = iter(b)
  same = True
  msg = []
  i = -1
  while True:
    i+=1
    try:
      al = genA.next()
    except StopIteration:
      try:
        genB.next() #see if B is done
        return False,msg + ['file '+str(b)+' has more lines thani '+str(a)]
      except StopIteration: #both are done
        return same,msg
    try:
      bl = genB.next()
    except StopIteration:
      return False,msg + ['file '+str(a)+' has more lines than '+str(b)]
    if al != bl:
      print 'not the same:\n'+al+bl
      same = False
      msg += ['line '+str(i)+' is not the same!']

#first test XML to XML
tree = TS.parse(file(os.path.join('parse','example_xml.xml'),'r'))
strTree = TS.tostring(tree)
xmlToXmlFileName = os.path.join('parse','fromXmltoXML.xml')
file(xmlToXmlFileName,'w').write(strTree)
same,msg = checkSameFile(file(xmlToXmlFileName,'r'),file(os.path.join('gold',xmlToXmlFileName),'r'))
if same:
  results['passed']+=1
else:
  results['failed']+=1
  print 'Failures in XML to XML:'
  for m in msg:
    print '  ',msg


#second test GetPot to XML
getpot = file(os.path.join('parse','example_getpot.i'),'r')
tree = TS.parse(getpot,dType='GetPot')
strTree = TS.tostring(tree)
getpotToXmlFileName = os.path.join('parse','fromGetpotToXml.xml')
file(getpotToXmlFileName,'w').write(strTree)
same,msg = checkSameFile(file(getpotToXmlFileName,'r'),file(os.path.join('gold',getpotToXmlFileName),'r'))
if same:
  results['passed']+=1
else:
  results['failed']+=1
  print 'Failures in GetPot to XML:'
  for m in msg:
    print '  ',msg


print 'Results:', list('%s: %i' %(k,v) for k,v in results.items())
sys.exit(results['failed'])
