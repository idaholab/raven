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
from collections import OrderedDict
from UQTemplate.UQTemplate import UQTemplate

temp = UQTemplate()

temp.loadTemplate('uq_template.xml', os.path.dirname(__file__))

# information needed by UQTemplate to make run file
model = {'file': '../../PostProcessors/BasicStatistics/simpleMirrowModel',
         'output': ['x1'],
        }
variables = OrderedDict()
variables['x']= {'mean':100,
                  'std':50}
variables['y'] = {'mean':100,
                   'std':50}

case = 'sample_mirrow'
numSamples = 100
workflow = 'UQTemplate/new_uq.xml'

template = temp.createWorkflow(model=model, variables=variables, samples=numSamples, case=case)
temp.writeWorkflow(template, workflow, run=True)
# cleanup?
