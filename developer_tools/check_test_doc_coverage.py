import xml.etree.ElementTree as ET
import get_coverage_tests as gct
import os

tests = gct.getRegressionTests()

needDirs = []
for folder,files in tests.items():
  print 'checking dir',folder
  ok = []
  bad = []
  for f in files:
    root = ET.parse(os.path.join(folder,f)).getroot()
    if root.find('TestInfo') is not None:
      ok.append(f)
    else:
      bad.append(f)
  if len(bad) > 0: needDirs.append(folder)
  print '  not documented:'
  for f in bad:
    print '    ',f

print ''
print 'Summary of Folders that Need Attention:'
for f in needDirs:
  print '  ',f



