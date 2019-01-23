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
import sys

infile = sys.argv[1]

for line in open(infile,'r'):
  if line.startswith('x ='):
    x=float(line.split('=')[1])
  if line.startswith('y ='):
    y=float(line.split('=')[1])
  if line.startswith('out ='):
    out=line.split('=')[1].strip()

# generate fails roughly half the time.
if x+y>0:
  raise RuntimeError('Answer is bigger than 0.  Just a test error.')

outfile = open(out+'.csv','w')
outfile.writelines('x,y,ans\n')
outfile.writelines(','.join([str(x),str(y),str(x+y)]))
outfile.close()
