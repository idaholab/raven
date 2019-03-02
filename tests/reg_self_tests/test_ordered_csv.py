"""
This module performs unit tests for the OrderedCSVDiffer
"""

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import sys
import os

my_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
base_dir = os.path.dirname(os.path.dirname(my_dir))

test_system_dir = os.path.join(base_dir,"scripts","TestHarness","testers")
print(test_system_dir)
sys.path.append(test_system_dir)
rook_system_dir = os.path.join(base_dir,"rook")
sys.path.append(rook_system_dir)

from OrderedCSVDiffer import OrderedCSVDiffer

print(OrderedCSVDiffer)

results = {"pass":0,"fail":0}

for expected_same, filename in [(True,"same.csv"),
                                (True,"col_swap.csv"),
                                (False,"diff_len.csv"),
                                (False,"diff_num.csv"),
                                (False,"diff_word.csv"),
                                (False,"diff_col_word.csv"),
                                (False,"diff_type.csv")]:
  orig = os.path.join(my_dir, filename)
  gold = os.path.join(my_dir, "gold", filename)
  differ = OrderedCSVDiffer([filename], [gold])
  same, message = differ.diff()
  print(message)
  if expected_same != same:
    print("Expected:",expected_same,"Got",same,"Message",message)
    results["fail"] += 1
  else:
    results["pass"] += 1


print(results)
sys.exit(results["fail"])
