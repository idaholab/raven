#
# Dummy converter just to standardize XML inputs.
#
import convert_utils
import sys

def convert(tree):
  return tree

if __name__=='__main__':
  convert_utils.standardMain(sys.argv,convert)
