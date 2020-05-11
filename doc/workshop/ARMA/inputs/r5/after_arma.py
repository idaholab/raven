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
# All answers are 1.0

import numpy as np

def run(self,Input):
  wEng=[]
  fullPower = 0.5 * 0.35 * 1.17682 * (14**3) * np.pi * (58.13**2) / 4 / 1000000

  for j in range(len(self.Time)):
    speed = self.Speed[j]
    load = self.Load[j]
    if speed < 3 or speed > 25:
      eng = 0
    elif speed > 14:
      eng = fullPower
    else:
      eng = fullPower * (speed**3) / (14**3)
    eng = min(eng*self.turbinenumber,load)
    wEng.append(eng)

  totalLoad = np.sum(self.Load)
  totalWindEng = np.sum(wEng)
  totalWindFullP = fullPower*self.turbinenumber*len(self.Time)
  self.CapF = totalWindEng / totalWindFullP

  self.engDiff = float(totalLoad - totalWindEng) / len(self.Time)
