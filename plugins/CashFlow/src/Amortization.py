import numpy as np

# AMORTAZIATION SCHEMES


MACRS = { 20: 0.01 * np.array([3.750, 7.219, 6.677, 6.177, 5.713, 5.285, 4.888, 4.522, 4.462 , 4.461, 4.462, \
          4.461, 4.462, 4.461, 4.462, 4.461, 4.462, 4.461, 4.462, 4.461, 2.231]),
          15: 0.01 * np.array([5.0, 9.5, 8.55, 7.7, 6.93, 6.23, 5.9, 5.9, 5.91, 5.9, 5.91, 5.9, 5.91, 5.9, 5.91, 2.95]),
          10: 0.01 * np.array([10.00, 18.00, 14.40, 11.52, 9.22, 7.37, 6.55, 6.55, 6.56, 6.55, 3.28]),
          7: 0.01 * np.array([14.29, 24.49, 17.49, 12.49, 8.93, 8.92, 8.93, 4.46]),
          5: 0.01 * np.array([20.00, 32.00, 19.20, 11.52, 11.52, 5.76]),
          3: 0.01 * np.array([33.33, 44.45, 14.81, 7.41])}

def amortize(scheme, plan, start_value, component_life):
  """
    return the amortization plan
    @ In, scheme, str, 'macrs' or 'custom'
    @ In, plan, list or array like, list of provided MACRS values
    @ In, start_value, float, the given initial Capex value
    @ In, component_life, int, the life of component
    @ Out, alpha, numpy.array, array of alpha values for given scheme
  """
  alpha = np.zeros(component_life + 1, dtype=float)
  lscheme = scheme.lower()
  if lscheme == 'macrs':
    ys = plan[0]
    pcts = MACRS.get(ys, None)
    if pcts is None:
      raise IOError('Provided MACRS "{}" is not allowed.'.format(ys))
    alpha[1:len(pcts)+1] = pcts * start_value
  elif lscheme == 'custom':
    alpha[1:len(plan)+1] = np.asarray(plan)/100. * start_value
  else:
    raise NotImplementedError('Amortization scheme "{}" not yet implemented.'.format(scheme))
  return alpha
