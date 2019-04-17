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
"""
Created on February 22 2019
@author: talbpaul

Template interface for the test UQ template input.
"""
from __future__ import print_function, unicode_literals
import os
import configparser
from collections import OrderedDict
from UQTemplate.UQTemplate import UQTemplate

print('Loading template ...')
temp = UQTemplate()
temp.loadTemplate('uq_template.xml', os.path.dirname(__file__))
print(' ... template loaded')

# information needed by UQTemplate to make run file
print('Loading input file ...')
config = configparser.ConfigParser()
config.read('UQTemplate/uq_template_input.i')
print(' ... input file loaded')


model = {'file': config.get('model', 'file'),
         'output': list(x.strip() for x in config.get('model', 'output').split(','))
        }

variables = OrderedDict()
for var in config['variables'].keys():
  mean, std = list(float(x) for x in config.get('variables', 'x').split(','))
  variables[var] = {'mean': mean,
                    'std': std}

case = config.get('settings', 'case')
numSamples = config.getint('settings', 'samples')
workflow = os.path.join('UQTemplate', config.get('settings', 'workflow'))

print('Writing RAVEN file ...')
template = temp.createWorkflow(model=model, variables=variables, samples=numSamples, case=case)
errors = temp.writeWorkflow(template, workflow, run=False)

# finish up
if errors == 0:
  print('\n\nSuccessfully wrote input "{}". Run it with RAVEN!\n'.format(workflow))
else:
  print('\n\nProblems occurred while running the code. See above.\n')
