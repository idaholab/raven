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
  Generates histories to be read in and used as training for LSTM model
  Uses example found in https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
"""
import numpy as np
import xarray as xr

alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphaMap = dict(zip(alpha, range(len(alpha))))

np.random.seed(42)

numTraining = 100
numTesting = 10
numSamples = numTraining + numTesting
seqLen = 3
seq = np.asarray(range(seqLen))
samples = np.zeros((numSamples, seqLen))
outputs = np.zeros((numSamples))
initials = np.random.randint(0, high=len(alpha)-seqLen, size=numSamples)
t = 0.1 * seq
for n in range(numSamples):
  x0 = initials[n]
  samples[n, :] = x0 + seq
  outputs[n] = x0 + seqLen

# training
ds = xr.Dataset(data_vars={'presequence': (['RAVEN_sample_ID', 't'], samples[:numTraining]),
                           'nextval': (['RAVEN_sample_ID'], outputs[:numTraining])},
                coords={'RAVEN_sample_ID': range(numTraining),
                        't': t}
               )
df = ds.to_dataframe()
df.to_csv('training.csv', mode='w', header=True)

# testing
ds = xr.Dataset(data_vars={'presequence': (['RAVEN_sample_ID', 't'], samples[numTraining:]),
                           'nextval': (['RAVEN_sample_ID'], outputs[numTraining:])},
                coords={'RAVEN_sample_ID': range(numTesting),
                        't': t}
               )
df = ds.to_dataframe()
df.to_csv('testing.csv', mode='w', header=True)
