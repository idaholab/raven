#
# Dummy just for standardization
#
import sys
import convert_utils

def convert(tree,fileName=None):
  """Does nothing but return the tree.
  @ In, tree, XMLtree.
  @ In, fileName, the name for the raven input file
  @ Out, XMLtree, same tree.
  """
  return tree

if __name__=='__main__':
  convert_utils.standardMain(sys.argv,convert)
