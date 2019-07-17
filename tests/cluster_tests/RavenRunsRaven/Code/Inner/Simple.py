import sys
import argparse
import configparser

def check_aux():
  try:
    open('simple.aux', 'r')
  except FileNotFoundError:
    raise RuntimeError('Aux file not found prior to running!')

def get_input():
  parser = argparse.ArgumentParser(description='RrR Test Code')
  parser.add_argument('-i', type=str, help='main input file')
  args = parser.parse_args()
  return args

def read_input(infile):
  config = configparser.ConfigParser()
  config.read(infile)
  a = float(config['FromOuter']['a'])
  b = float(config['FromOuter']['b'])
  x = float(config['FromInner']['x'])
  y = float(config['FromInner']['y'])
  out = config['Output']['output']
  return a, b, x, y, out

def run(*args):
  return sum(args)

def write(a, b, c, x, y, out):
  print('Writing to', out)
  with open(out, 'w') as f:
    f.writelines(','.join('abcxy') + '\n')
    f.writelines(','.join(str(i) for i in [a, b, c, x, y]) + '\n')

if __name__ == '__main__':
  check_aux()
  args = get_input()
  infile_name = args.i
  a, b, x, y, out = read_input(infile_name)
  c = run(a, b, x, y)
  write(a, b, c, x, y, out)
