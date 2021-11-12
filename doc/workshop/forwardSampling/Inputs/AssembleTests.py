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
import copy
frameworkPath = os.path.join(os.path.dirname(__file__), *['..']*4, 'framework')
sys.path.append(frameworkPath)
from utils import xmlUtils

def write1():
  """
    Writes the corresponding test.
    @ In, None
    @ Out, None
  """
  name = '1_sample_and_plot.xml'
  root, _ = xmlUtils.loadToTree('../exercises/{}'.format(name))
  # remove interactive plot
  xmlUtils.findPath(root, 'OutStreams/Plot/actions/how').text = 'png'
  # write
  xmlUtils.toFile('test_'+name, root)

def write2():
  """
    Writes the corresponding test.
    @ In, None
    @ Out, None
  """
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
  """
    Writes the corresponding test.
    @ In, None
    @ Out, None
  """
  fromName = 'test_2_normal_distribution.xml'
  toName = '3_initial_height.xml'
  root, _ = xmlUtils.loadToTree(fromName)
  # workdir
  xmlUtils.findPath(root, 'RunInfo/WorkingDir').text = 'r3'
  # change v0 to y0
  ## models
  xmlUtils.findPath(root, 'Models/ExternalModel/variables').text += ',y0'
  ## samplers
  samp = xmlUtils.findPath(root, 'Samplers/MonteCarlo')
  samp.remove(xmlUtils.findPath(samp, 'constant[@name=\'y0\']'))
  var = xmlUtils.findPath(samp, 'variable[@name=\'v0\']')
  var.attrib['name'] = 'y0'
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
  xmlUtils.findPath(root, 'DataObjects/PointSet[@name=\'placeholder\']/Input').text = 'y0,angle'
  xmlUtils.findPath(root, 'DataObjects/PointSet[@name=\'results\']/Input').text = 'y0,angle'
  ## plot
  xmlUtils.findPath(root, 'OutStreams/Plot/plotSettings/plot/x').text = 'results|Input|y0'
  xmlUtils.findPath(root, 'OutStreams/Plot/plotSettings/xlabel').text = 'y0'
  # write
  xmlUtils.toFile('test_'+toName, root)

def write4():
  """
    Writes the corresponding test.
    @ In, None
    @ Out, None
  """
  fromName = 'test_3_initial_height.xml'
  toName = '4_grid_sampler.xml'
  root, _ = xmlUtils.loadToTree(fromName)
  # workdir
  xmlUtils.findPath(root, 'RunInfo/WorkingDir').text = 'r4'
  # Grid
  grid = copy.deepcopy(xmlUtils.findPath(root, 'Samplers/MonteCarlo'))
  root.find('Samplers').append(grid)
  grid.tag = 'Grid'
  grid.attrib['name'] = 'my_grid'
  grid.remove(grid.find('samplerInit'))
  dist = xmlUtils.newNode('grid', attrib={'construction':'equal', 'steps':15, 'type':'CDF'}, text='0.0 1.0')
  for var in grid.findall('variable'):
    var.append(dist)
  # steps
  xmlUtils.findPath(root, 'Steps/MultiRun/Sampler').attrib['type'] = 'Grid'
  xmlUtils.findPath(root, 'Steps/MultiRun/Sampler').text = 'my_grid'
  # write
  xmlUtils.toFile('test_'+toName, root)

def write5():
  """
    Writes the corresponding test.
    @ In, None
    @ Out, None
  """
  fromName = 'test_4_grid_sampler.xml'
  toName = '5_basic_stats.xml'
  root, _ = xmlUtils.loadToTree(fromName)
  # workdir, sequence
  xmlUtils.findPath(root, 'RunInfo/WorkingDir').text = 'r5'
  xmlUtils.findPath(root, 'RunInfo/Sequence').text += ',stats'
  # step
  step = xmlUtils.newNode('PostProcess', attrib={'name':'stats'})
  root.find('Steps').append(step)
  step.append(xmlUtils.newNode('Input', attrib={'class':'DataObjects', 'type':'PointSet'}, text='results'))
  step.append(xmlUtils.newNode('Model', attrib={'class':'Models', 'type':'PostProcessor'}, text='stats_pp'))
  step.append(xmlUtils.newNode('Output', attrib={'class':'DataObjects', 'type':'PointSet'}, text='stats_data'))
  step.append(xmlUtils.newNode('Output', attrib={'class':'OutStreams', 'type':'Print'}, text='stats_file'))
  # postprocessor
  pp = xmlUtils.newNode('PostProcessor', attrib={'name':'stats_pp', 'subType':'BasicStatistics'})
  root.find('Models').append(pp)
  pp.append(xmlUtils.newNode('expectedValue', attrib={'prefix':'mean'}, text='r,t'))
  pp.append(xmlUtils.newNode('variance', attrib={'prefix':'var'}, text='r,t'))
  sens = xmlUtils.newNode('sensitivity', attrib={'prefix':'sens'})
  pp.append(sens)
  sens.append(xmlUtils.newNode('features', text='r'))
  sens.append(xmlUtils.newNode('targets', text='y0,angle'))
  # data object
  data = xmlUtils.newNode('PointSet', attrib={'name':'stats_data'})
  root.find('DataObjects').append(data)
  data.append(xmlUtils.newNode('Output', text='mean_r,var_r,mean_t,var_t,sens_y0_r,sens_angle_r'))
  # out stream
  print = xmlUtils.newNode('Print', attrib={'name':'stats_file'})
  root.find('OutStreams').append(print)
  print.append(xmlUtils.newNode('type', text='csv'))
  print.append(xmlUtils.newNode('source', text='stats_data'))
  # write
  xmlUtils.toFile('test_'+toName, root)

def write6():
  """
    Writes the corresponding test.
    @ In, None
    @ Out, None
  """
  fromName = '../exercises/6_point_sets.xml'
  toName = '6_point_sets.xml'
  root, _ = xmlUtils.loadToTree(fromName)
  # write
  xmlUtils.toFile('test_'+toName, root)

def write7():
  """
    Writes the corresponding test.
    @ In, None
    @ Out, None
  """
  fromName = 'test_6_point_sets.xml'
  toName = '7_history_sets.xml'
  root, _ = xmlUtils.loadToTree(fromName)
  # workdir
  xmlUtils.findPath(root, 'RunInfo/WorkingDir').text = 'r7'
  # step
  xmlUtils.findPath(root, 'Steps/MultiRun/Output').attrib['type'] = 'HistorySet'
  xmlUtils.findPath(root, 'Steps/IOStep/Input').attrib['type'] = 'HistorySet'
  # model
  xmlUtils.findPath(root, 'Models/ExternalModel/variables').text = 'v0,y0,angle,timeOption,x,y,t'
  # data object
  data = xmlUtils.findPath(root, 'DataObjects/PointSet[@name=\'results\']')
  data.tag = 'HistorySet'
  data.find('Output').text = 'x,y,t'
  opts = xmlUtils.newNode('options')
  data.append(opts)
  opts.append(xmlUtils.newNode('pivotParameter', text='t'))
  # write
  xmlUtils.toFile('test_'+toName, root)

def write8():
  """
    Writes the corresponding test.
    @ In, None
    @ Out, None
  """
  fromName = '../exercises/8_soln.xml'
  toName = '8_history_sync.xml'
  root, _ = xmlUtils.loadToTree(fromName)
  # write
  xmlUtils.toFile('test_'+toName, root)

def write9():
  fromName = 'test_8_history_sync.xml'
  toName = '9_time_stats.xml'
  root, _ = xmlUtils.loadToTree(fromName)
  # workdir
  xmlUtils.findPath(root, 'RunInfo/WorkingDir').text = 'r9'
  xmlUtils.findPath(root, 'RunInfo/Sequence').text += ',stats'
  # step
  step = xmlUtils.newNode('PostProcess', attrib={'name':'stats'})
  root.find('Steps').append(step)
  step.append(xmlUtils.newNode('Input', attrib={'class':'DataObjects', 'type':'HistorySet'}, text='synced'))
  step.append(xmlUtils.newNode('Model', attrib={'class':'Models', 'type':'PostProcessor'}, text='stats_pp'))
  step.append(xmlUtils.newNode('Output', attrib={'class':'DataObjects', 'type':'HistorySet'}, text='stats_data'))
  step.append(xmlUtils.newNode('Output', attrib={'class':'OutStreams', 'type':'Print'}, text='stats_file'))
  # postprocessor
  pp = xmlUtils.newNode('PostProcessor', attrib={'name':'stats_pp', 'subType':'BasicStatistics'})
  root.find('Models').append(pp)
  pp.append(xmlUtils.newNode('pivotParameter', text='t'))
  pp.append(xmlUtils.newNode('expectedValue', attrib={'prefix':'mean'}, text='x,y'))
  pp.append(xmlUtils.newNode('variance', attrib={'prefix':'var'}, text='x,y'))
  sens = xmlUtils.newNode('sensitivity', attrib={'prefix':'sens'})
  pp.append(sens)
  sens.append(xmlUtils.newNode('features', text='x,y'))
  sens.append(xmlUtils.newNode('targets', text='v0,y0,angle'))
  # data object
  data = xmlUtils.newNode('HistorySet', attrib={'name':'stats_data'})
  root.find('DataObjects').append(data)
  data.append(xmlUtils.newNode('Output', text='mean_x,var_x,mean_y,var_y,sens_y0_x,sens_angle_x,sens_v0_x,sens_y0_y,sens_angle_y,sens_v0_y'))
  opts = xmlUtils.newNode('options')
  data.append(opts)
  opts.append(xmlUtils.newNode('pivotParameter', text='t'))
  # out stream
  print = xmlUtils.newNode('Print', attrib={'name':'stats_file'})
  root.find('OutStreams').append(print)
  print.append(xmlUtils.newNode('type', text='csv'))
  print.append(xmlUtils.newNode('source', text='stats_data'))
  # write
  xmlUtils.toFile('test_'+toName, root)

if __name__ == '__main__':
  write1()
  write2()
  write3()
  write4()
  write5()
  write6()
  write7()
  write8()
  write9()
