# Copyright Nucube Energy, Inc.

def testSerpentStoppingCriteria(raven):
  if raven.impKeff_0[-1] < 1.0:
    return False
  return True
 
