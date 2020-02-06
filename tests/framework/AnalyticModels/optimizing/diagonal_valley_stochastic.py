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
import numpy as np
#static seed
np.random.seed(42)


def run(raven, Inputs):
  coeffs = [1, 1, 0] # ax + by + c = 0 -> x = y
  raven.ans = main(coeffs, raven.x, raven.y) + random(scale=raven.stoch)/10

def main(coeffs, x, y, thresh=0.01):
  distance = dist_to_line(coeffs, x, y)
  z = 3*(x+0.5)**2 + 3*(y-0.5)**2
  z += distance * 10
  return z

def dist_to_line(coeffs, x0, y0):
  cx, cy = closest_point(coeffs, x0, y0)
  dist = np.sqrt((x0 - cx)**2 + (y0 - cy)**2)
  return dist

def closest_point(coeffs, x0, y0):
  a, b, c = coeffs
  denom = a*a + b*b
  x = b * (b * x0 - a * y0) - a * c
  x /= denom
  y = a * (-b * x0 + a * y0) - b * c
  y /= denom
  return x, y

def random(scale=0.5,loc=-1.0):
  return scale*(2.*np.random.rand()+loc)

