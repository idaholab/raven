
from __future__ import division, print_function, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)


import os
import sys

#XXX fixme to find a better way to the tests directory

this_dir = os.path.abspath(os.path.dirname(__file__))
test_dir = os.path.join(os.path.dirname(this_dir),"tests")

print(this_dir,test_dir)

def get_test_lists(directory):
  """
  Returns a list of all the files named tests under the directory
  directory: the directory to start at
  """
  test_list = []
  for root, dirs, files in os.walk(test_dir):
    if 'tests' in files:
      test_list.append(os.path.join(root,'tests'))
  return test_list

print(get_test_lists(test_dir))
