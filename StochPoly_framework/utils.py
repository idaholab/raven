def toString(s):
  if type(s) == type(""):
    return s
  else:
    return s.decode()

def toBytes(s):
  if type(s) == type(""):
    return s.encode()
  else:
    return s
