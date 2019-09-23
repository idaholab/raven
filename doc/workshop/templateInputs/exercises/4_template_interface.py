import os
import sys
ProjectileTemplateClassModule = __import__('5_template_class')
ProjectileTemplateClass = ProjectileTemplateClassModule.ProjectileTemplateClass

def read_input(file_name):
  """ read in entries from the file named file_name """
  inputs = {}
  # open up the file for reading
  with open(file_name, 'r') as f:
    # iterate over the lines in the file
    for line in f:
      line = line.strip()            # remove whitespace on the left and right sides
      line = line.split('#')[0]      # remove comments
      if line == '': continue        # skip empty lines
      try:
        key, value = line.split('=') # get keys and values
      except ValueError:
        raise IOError('Line not in correct format of "key = value": {}'.format(line[:-1]))
      key = key.strip().lower()      # force everything to be lowercase so it's easier on the user
      value = value.strip().lower()
      inputs[key] = value            # store line entry
  return inputs

def interpret_input(inputs):
  """ convert input entries to usable dictionaries """
  for key, value in inputs.items():                   # interpret each line's worth of entries
    if key in ['v0', 'y0', 'angle']:                    # for variables, intepret distributions
      converted = interpret_distribution(key, value)  # use a separate method to keep things clean
    elif key == 'metric':                             # metrics are easy, they're just a list
      converted = list(x.strip().lower() for x in value.split(','))
      for c in converted:                             # check the metrics are valid entries
        if c not in ['mean', 'std', 'percentile']:
          raise IOError('Unrecognized metric:', c)
    else:
      raise IOError('Unrecognized keyword entry: {} = {}'.format(key, value))
    inputs[key] = converted                           # replace the key with the converted values
  return inputs

def interpret_distribution(key, value):
  """ interpret input distribution descriptions into usable entries """
  dist_name, args = value.split('(')                  # uniform(a, b) to ['Uniform', 'a, b)']
  dist_name = dist_name.strip()                       # clear whitespace
  args = args.strip(')').split(',')                   # 'a, b)' to ['a', 'b']
  converted = {}                                      # storage for distribution information
  if dist_name not in ['uniform', 'normal', 'constant']:
    raise IOError('Variable distributions can only be "uniform", "normal", or "constant"; '+
                  'got "{}" for "{}"'.format(dist_name, key))
  converted['dist'] = dist_name                       # always store the distribution type
  if dist_name == 'uniform':                          # interpret uniform distributions
    converted['lowerBound'] = float(args[0])
    converted['upperBound'] = float(args[1])
  elif dist_name == 'normal':                         # interpret normal distributions
    converted['mean'] = float(args[0])
    converted['std'] = float(args[1])
    if len(args) > 2:
      converted['lowerBound'] = float(args[2])        # note that normal distributions can be truncated
      converted['upperBound'] = float(args[3])
  elif dist_name == 'constant':                       # constants are simple!
    converted['value'] = float(args[0])
  return converted

if __name__ == '__main__':                            # this is how this file gets run
  print('Welcome to the Projectile Templated UQ RAVEN Runner!')
  if len(sys.argv) != 2:                              # read the input file from the call arguments, as "python <script> <inputfile>"
    raise IOError('Expected 1 argument for interface (the name of the input file)!')
  # read inputs
  file_name = sys.argv[1]
  inp = read_input(file_name)                         # load the input file
  inp = interpret_input(inp)                          # interpret the data
  # create template class instance
  templateClass = ProjectileTemplateClass()
  # load template
  here = os.path.abspath(os.path.dirname(__file__))
  print(' ... working directory:', here)
  templateClass.loadTemplate('2_soln_template.xml', here)
  print(' ... workflow successfully loaded ...')
  # create modified template
  template = templateClass.createWorkflow(inp)
  print(' ... workflow successfully modified ...')
  # write files
  templateClass.writeWorkflow(template, os.path.join(here, '6_ProjUQInput.xml'), run=True)
  print('')
  print(' ... workflow successfully created and run ...')
  print(' ... Complete!')
  print('This is the end of the Projectile Templated UQ Raven Runner. Have a good one!')

