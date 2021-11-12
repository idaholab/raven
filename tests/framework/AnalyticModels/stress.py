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

# Simply a high-load model to be run for an input amount of time.
# Expected input is "interval", which is the run length in seconds
# Output is "done" which always evaluates to 1.0
#
import time

def run(self,Input):
  start = time.time()
  while time.time() < start + self.interval:
    True
  self.done = 1.0
