import csv
import argparse

# Build Parser
parser = argparse.ArgumentParser(description="Reads several variables from input and performs simple calculation to produce output.")
parser.add_argument('-i', dest='input_file', required=True,
  help='Input file with variables a, b, c, d, and e', metavar='INFILE')
parser.add_argument('-o', dest='output_file', required=True,
  help='Output file name without .csv extension', metavar = 'OUTFILE')
args = parser.parse_args()

# Read values from INFILE
with open(args.input_file) as rf:
  indata = rf.readlines()
  for line in indata:
    # parse input, assign values to variables
    variable, value = line.split("=")
    exec('%s = %f' % (variable.strip(),float(value.strip())))
rf.close()

# Calculation
f = a*b
g = c/5 + d/3
h = g + a * e

# Print to csv file
out_file_name = args.output_file + '.csv'
print "Output will be printed to", out_file_name
with open(out_file_name, 'wb') as wf:
  writer = csv.writer(wf, delimiter=',')
  var_name = ['a','b','c','d','e','f','g','h']
  data = [a,b,c,d,e,f,g,h]
  writer.writerow(var_name)
  writer.writerow(data)
wf.close()
