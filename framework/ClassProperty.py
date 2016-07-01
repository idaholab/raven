"""
    This allows for the construction of static constant class properties that
    any object can use, but they should not be allowed to modify
"""

## Custom class allows for the @ClassProperty decorator to be used to specify
## a class level, immutable, and static property

class ClassProperty(object):
  """
      A custom class providing the option to create a class level decorator
      (e.g. @ClassProperty) above a method to make it a static and immutable
      class variable
  """
  def __init__(self, func):
    """
        Constructor that will associate this ClassProperty to a particular
        method
        @ In, func, a function
        @ Out, None
    """
    self.func = func

  def __get__(self, inst, cls):
    """
        Overloaded getter that will ignores the instance and returns the class
        call to the ClassProperty associated function
        @ In, inst, instance object attempting to retrieve this ClassProperty
          (ignored, thus it can be undefined)
        @ In, cls, class attempting to retrieve this ClassProperty
        @ Out, variable, this will return whatever the ClassProperty's func
          returns
    """
    return self.func(cls)