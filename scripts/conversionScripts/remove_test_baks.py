import os
import get_coverage_tests as gct

tests = gct.returnTests()
for t in tests:
  print 'Removing ',t
  os.system('rm %s.bak' %t)
