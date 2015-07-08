#
# Dummy just for standardization
#
import sys
import convert_utils

def convert(tree):
  """Does nothing but return the tree.
  @ In, tree, XMLtree.
  @ Out, XMLtree, same tree.
  """
  return tree

if __name__=='__main__':
  convert_utils.standardMain(sys.argv,convert)
