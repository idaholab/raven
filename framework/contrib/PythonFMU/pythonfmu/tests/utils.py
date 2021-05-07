from pythonfmu.variables import Boolean, Integer, Real, ScalarVariable, String

FMI2PY = dict((
    (Boolean, bool),
    (Integer, int),
    (Real, float),
    (String, str)
))
PY2FMI = dict([(v, k) for k, v in FMI2PY.items()])
