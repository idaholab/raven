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

from __future__ import print_function, unicode_literals, absolute_import
import sys

disclaimer = """# Copyright 2017 Battelle Energy Alliance, LLC
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
"""
firstLine = disclaimer.split('\n')[-1].strip()

def disclaimFile(filename):
  """
    Adds the standard BEA copyright and Apache license header to provided file.
    @ In, filename, str, file (including path and extension) to modify
    @ Out, None
  """
  # read in file contents
  with open(filename,'r') as f:
    contents = f.read()
  # check for contents -> note this might not work if file formatted with carriage returns besides \n
  found = disclaimer in contents
  # nothing to do if it's there
  if found:
    print('Disclaimer already present in',filename)
    return
  # otherwise, add it
  contents = disclaimer + contents
  with open(filename,'w') as f:
    f.write(contents)
  print('Disclaimer added to',filename)

if __name__=='__main__':
  targets = sys.argv[1:]
  for t in targets:
    disclaimFile(t)

