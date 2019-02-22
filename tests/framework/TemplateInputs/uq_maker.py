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
