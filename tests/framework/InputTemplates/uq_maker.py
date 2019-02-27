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
import os
import configparser
from collections import OrderedDict
from UQTemplate.UQTemplate import UQTemplate

temp = UQTemplate()

temp.loadTemplate('uq_template.xml', os.path.dirname(__file__))

# information needed by UQTemplate to make run file
config = configparser.ConfigParser()
config.read('UQTemplate/uq_template_input.i')


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

template = temp.createWorkflow(model=model, variables=variables, samples=numSamples, case=case)
temp.writeWorkflow(template, workflow, run=True)

# finish up
print('\n\nSuccessfully performed uncertainty quantification. See results in UQTemplate/{}/stats.csv\n'.format(case))
