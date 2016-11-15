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
      same = False
      msg += ['line '+str(i)+' is not the same!']



#first test XML to XML
print 'Testing XML to XML ...'
tree = TS.parse(file(os.path.join('parse','example_xml.xml'),'r'))
strTree = TS.tostring(tree)
fname = os.path.join('parse','fromXmltoXML.xml')
file(fname,'w').write(strTree)
same,msg = checkSameFile(file(fname,'r'),file(os.path.join('gold',fname),'r'))
if same:
  results['passed']+=1
  print '  ... passed!'
else:
  results['failed']+=1
  print '  ... failures in XML to XML:'
  print '     ',msg[0]



getpot = file(os.path.join('parse','example_getpot.i'),'r')
gtree = TS.parse(getpot,dType='GetPot')
#third test GetPot to XML
print 'Testing GetPot to XML ...'
strTree = TS.tostring(gtree)
fname = os.path.join('parse','fromGetpotToXml.xml')
file(fname,'w').write(strTree)
same,msg = checkSameFile(file(fname,'r'),file(os.path.join('gold',fname),'r'))
if same:
  results['passed']+=1
  print '  ... passed!'
else:
  results['failed']+=1
  print '  ... failures in GetPot to XML:'
  print '     ',msg[0]


#finally test XML to GetPot
print 'Testing XML to GetPot ...'
strTree = tree.printGetPot()
fname = os.path.join('parse','fromXmltoGetpot.i')
file(fname,'w').write(strTree)
same,msg = checkSameFile(file(fname,'r'),file(os.path.join('gold',fname),'r'))
if same:
  results['passed']+=1
  print '  ... passed!'
else:
  results['failed']+=1
  print '  ... failures in GetPot to XML:'
  print '     ',msg[0]


#second test Getpot to GetPot
print 'Testing GetPot to GetPot ...'
getpot = file(os.path.join('parse','example_getpot.i'),'r')
gtree = TS.parse(getpot,dType='GetPot')
strTree = gtree.printGetPot()
fname = os.path.join('parse','fromGetpotToGetpot.i')
file(fname,'w').write(strTree)
same,msg = checkSameFile(file(fname,'r'),file(os.path.join('gold',fname),'r'))
if same:
  results['passed']+=1
  print '  ... passed!'
else:
  results['failed']+=1
  print '  ... failures in GetPot to Getpot:'
  print '     ',msg[0]








print 'Results:', list('%s: %i' %(k,v) for k,v in results.items())
sys.exit(results['failed'])
