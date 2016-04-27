import os,sys

pathToGCT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),'developer_tools')
sys.path.append(pathToGCT)
import get_coverage_tests as gct

tests = gct.getRegressionList()
for t in tests:
  print 'Removing ',t
  os.system('rm %s.bak' %t)
