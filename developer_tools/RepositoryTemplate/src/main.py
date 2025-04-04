"""
Created on October 14, 2024
@author: wangc
"""
import os
import sys
import logging
import argparse

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
# To enable the logging to both file and console, the logger for the main should be the root,
# otherwise, a function to add the file handler and stream handler need to be created and called by each module.
# logger = logging.getLogger(__name__)
logger = logging.getLogger()
# # create file handler which logs debug messages
fh = logging.FileHandler(filename='out.log', mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)


def main():
  logger.info('Welcome!')
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-i', '--input', nargs=1, required=True, help='Input Filename')
  parser.add_argument('-o', '--output', nargs=1, help='Output Filename')

  args = parser.parse_args()
  args = vars(args)
  inFile = args['input'][0]
  baseName = os.path.basename(inFile)
  workDir = os.path.dirname(os.path.abspath(os.path.normpath(inFile)))
  logger.info('Input file: %s', inFile)
  if args['output'] is not None:
    outFile = args['output'][0]
    logger.info('Output file: %s', outFile)
  else:
    outFile = os.path.join(workDir, 'out_' + baseName)
    logger.warning('Output file is not specifies, default output file with name ' + outFile + ' will be used')

  logger.info(' ... Complete!')

if __name__ == '__main__':
  sys.exit(main())
