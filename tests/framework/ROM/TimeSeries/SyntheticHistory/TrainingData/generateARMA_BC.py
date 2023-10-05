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
  Creates additional ARMA training data for interpolated ROMs
"""
import pandas as pd

# Load up the already-generated ARMA_A_0.csv
arma = pd.read_csv('ARMA_A_0.csv', index_col=0)

# Split the signal0 and signal1 columns into separate files
signal0 = arma[['signal0']]
signal0.to_csv('ARMA_B_0.csv')

signal1 = arma[['signal1']]
signal1 = signal1.rename(columns={'signal1': 'signal0'})
signal1.to_csv('ARMA_C_0.csv')

pointerCSV = pd.DataFrame({
  'scaling': [1, 1],
  'macro': [1, 3],
  'filename': ['ARMA_B_0.csv', 'ARMA_C_0.csv']
})
pointerCSV.to_csv('ARMAInterpolated.csv', index=False)
