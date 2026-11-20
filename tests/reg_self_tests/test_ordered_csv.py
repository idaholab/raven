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

# check the ignored_columns option: (expected_same, filename, ignored columns, description)
for expected_same, filename, ignored, description in [
    (False, "diff_num.csv", None, "column z differs, nothing ignored"),
    (True, "diff_num.csv", ["z"], "column z differs, z is ignored"),
    (False, "diff_num.csv", ["x"], "column z differs, another column is ignored"),
    (False, "diff_num.csv", ["x", "y"], "column z differs, two other columns are ignored"),
    (True, "diff_num.csv", ["x", "z"], "column z differs, z and another column are ignored"),
    (True, "diff_num.csv", ["not_a_column", "z"], "a name that is not a column is ignored"),
    (False, "diff_col_word.csv", None, "column z has words instead of numbers, nothing ignored"),
    (True, "diff_col_word.csv", ["z"], "column z has words instead of numbers, z is ignored"),
    (False, "diff_len.csv", ["z"], "different number of rows, a column is ignored"),
    (False, "extra_col.csv", None, "each file has a column the other lacks, nothing ignored"),
    (False, "extra_col.csv", ["z"], "each file has a column the other lacks, only one is ignored"),
    (True, "extra_col.csv", ["z", "extra"], "each file has a column the other lacks, both are ignored")]:
  gold = os.path.join(my_dir, "gold", filename)
  differ = OrderedCSVDiffer([os.path.join(my_dir, filename)], [gold], ignored_columns=ignored)
  same, message = differ.diff()
  print(description, "->", same)
  if expected_same != same:
    print("Expected:",expected_same,"Got",same,"for",description,"Message",message)
    results["fail"] += 1
  else:
    results["pass"] += 1


print(results)
sys.exit(results["fail"])
