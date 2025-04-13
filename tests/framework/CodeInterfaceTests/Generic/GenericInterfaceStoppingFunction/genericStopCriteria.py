# Copyright Nucube Energy, Inc.

def stoppingCriteria(raven):
  if raven.poly[-1] > 0.2:
    return False
  return True

