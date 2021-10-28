import os
import sys
import matplotlib.pyplot as plt

frameworkDir='/Users/mandd/projects/raven/framework/'
sys.path.append(frameworkDir)
from Driver import ravenCaller

targetWorkflow = 'basic.xml'

ravenInstance = ravenCaller(frameworkDir)
ravenInstance.loadWorkflowFromFile(targetWorkflow)
ravenInstance.run()
results = ravenInstance.getEntity('DataObjects', 'results')
data = results.asDataset() 

data.plot.scatter(x="v0", y="angle", hue="r")
plt.show()
