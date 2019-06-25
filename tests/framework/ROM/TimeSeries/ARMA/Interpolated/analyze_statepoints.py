import pickle as pk
import pandas as pd
import numpy as np

pd.set_option('line_width', 150)

frames = []
for s in ['global'] + list(range(10)):
  print('SEGMENT:', s)
  # load statepoint years
  with open('debug_statepoints_{}.pk'.format(s), 'rb') as f:
    df = pk.load(f)
  df['interped'] = np.asarray([False, False])
  # load interpolated years
  for y in range(1,9):
    with open('debugg_interp_y{}_s{}.pk'.format(y, s), 'rb') as f:
      params = pk.load(f)
    params = dict((param, np.atleast_1d(value)) for param, value in params.items())
    params['year'] = y
    params['interped'] = np.atleast_1d(True)
    params = df.from_dict(params)
    params = params.set_index('year')
    df = df.append(params)
  df = df.sort_index()
  frames.append(df)
  print(df)
  print('\n\n\n')


