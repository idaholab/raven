# Copyright 2025 NuCube Energy and Battelle Energy Alliance, LLC
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
  Created on Mar 27, 2025
  @author: Andrea Alfonsi
  
  This script is aimed to Run HTPIPE sending commands directly via command line (for the execution of the input)
  
"""```
if __name__=='__main__':
  import sys
  import subprocess
  args = sys.argv
  executable = args[args.index('-e')+1] if '-e' in args else None
  inputFileName = args[args.index('-i')+1] if '-i' in args else None
  cmd = [executable,]
  p = subprocess.run(cmd, input=f"2\n{inputFileName}\nn\n".encode(), stdout=subprocess.PIPE)
  print(p.stdout.decode())
