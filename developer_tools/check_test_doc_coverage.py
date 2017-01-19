import xml.etree.ElementTree as ET
import get_coverage_tests as gct
import os

tests = gct.getRegressionTests()

needDirs = []
for folder,files in tests.items():
  print 'checking dir',folder
  ok = []
  bad = []
  error = []
  for f in files:
    try:
      root = ET.parse(os.path.join(folder,f)).getroot()
      if root.find('TestInfo') is not None:
        ok.append(f)
      else:
        bad.append(f)
    except: 
        error.append(f)

  if (len(bad) > 0) or (len(error) > 0): 
    needDirs.append(folder)

  if len(bad) > 0: 
    print '  not documented:'
    for f in bad:
      print '    ',f

  if len(error) > 0: 
    print '  XML Parse Errors Found:'
    for f in error:
      print '    ',f

print ''
print 'Summary of Folders that Need Attention:'
for f in needDirs:
  print '  ',f



