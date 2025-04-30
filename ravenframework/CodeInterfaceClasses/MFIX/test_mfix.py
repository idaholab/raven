from mfix_raven_interface import MFIX
import pickle
import matplotlib.pyplot as plt

workingDir = '/Users/wangc/projects/mfix/model_outputs'
command = ''
output = ''

mfix = MFIX()

output = mfix.finalizeCodeOutput(command, output, workingDir)

dataSet = mfix._dataSet

# dataSet.plot.scatter(x='time', y='height', z='void_frac', hue='void_frac')
# plt.show()

with open('dataset.pkl', 'wb') as f:
  pickle.dump(dataSet, f, protocol=-1)
