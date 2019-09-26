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

import os
import sys
frameworkPath = os.path.join(os.path.dirname(__file__), *['..']*4, 'framework')
sys.path.append(frameworkPath)
from utils import xmlUtils

def write1():
  name = '1_sample_and_plot.xml'
  root, _ = xmlUtils.loadToTree('../exercises/{}'.format(name))
  # remove interactive plot
  xmlUtils.findPath(root, 'OutStreams/Plot/actions/how').text = 'png'
  # write
  xmlUtils.toFile('test_'+name, root)

def write2():
  name = '2_normal_distribution.xml'
  root, _ = xmlUtils.loadToTree('../exercises/{}'.format(name))
  # add normal distribution node
  norm = xmlUtils.newNode('Normal', attrib={'name': 'vel_dist'})
  norm.append(xmlUtils.newNode('mean', text=30))
  norm.append(xmlUtils.newNode('sigma', text=5))
  root.find('Distributions').append(norm)
  # remove interactive plot
  xmlUtils.findPath(root, 'OutStreams/Plot/actions/how').text = 'png'
  # write
  xmlUtils.toFile('test_'+name, root)

def write3():
  fromName = 'test_2_normal_distribution.xml'
  toName = '3_initial_height.xml'
  root, _ = xmlUtils.loadToTree(fromName)
  # change v0 to y0
  ## models
  xmlUtils.findPath(root, 'Models/ExternalModel/variables').text += ',y0'
  ## samplers
  samp = xmlUtils.findPath(root, 'Samplers/MonteCarlo')
  print(samp)
  var = xmlUtils.findPath(samp, 'variable@name:v0').attrib['name'] = 'y0'
  var.find('distribution').text = 'y0_dist'
  samp.append(xmlUtils.newNode('constant', attrib={'name': 'v0'}, text=30))
  ## distributions
  distrs = root.find('Distributions')
  distrs.remove(distrs.find('Normal'))
  new = xmlUtils.newNode('Uniform', attrib={'name': 'y0_dist'})
  distrs.append(new)
  new.append(xmlUtils.newNode('lowerBound', text=0))
  new.append(xmlUtils.newNode('upperBound', text=1))
  ## data objects
  xmlUtils.findPath(root, 'DataObjects/PointSet@name:placeholder/Input').text = 'y0,angle'
  xmlUtils.findPath(root, 'DataObjects/PointSet@name:restuls/Input').text = 'y0,angle'
  ## plot
  xmlUtils.findPath(root, 'OutStreams/Plot/plotSettings/plot/x').text = 'results|Input|y0'
  xmlUtils.findPath(root, 'OutStreams/Plot/plotSettings/xlabel').text = 'y0'
  # write
  xmlUtils.toFile('test_'+toName, root)


if __name__ == '__main__':
  write1()
  write2()
  write3()
