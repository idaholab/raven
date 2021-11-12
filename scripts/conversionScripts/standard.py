# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Dummy just for standardization
#
import sys
import convert_utils

def convert(tree,fileName=None):
  """
    Does nothing but return the tree.
    @ In, tree, XMLtree.
    @ In, fileName, the name for the raven input file
    @ Out, XMLtree, same tree.
  """
  return tree

if __name__=='__main__':
  convert_utils.standardMain(sys.argv,convert)
