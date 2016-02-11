import os
import get_coverage_tests as gct

tests = gct.getRegressionList()
for t in tests:
  print 'Removing ',t
  os.system('rm %s.bak' %t)
