from options import *

import raven_check

test = {
  INPUT : 'TMI_test_PRA_transient_less_w.i',
  EXODIFF : ['TMI_test_PRA_transient_less_w_out_displaced.e'],
  REL_ERR : 1,
#SKIP: 'Unstable'
}

if not raven_check.has_python3:
    test[SKIP] = 'No python3 found' 

if not raven_check.has_swig2:
    test[SKIP] = 'No swig 2.0 found'

