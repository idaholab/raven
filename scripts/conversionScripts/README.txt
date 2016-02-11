Conversion scripts for maintaining input files.

To modify a single input file ('test.xml'), navigate to the directory
containing the test and run

python path/to/conversion/script.py test.xml

To run the conversion script on all the regression tests currently in
RAVEN, from anywhere run

python path/to/conversion/script.py --tests

To simply assure a file ('text.xml') is in the official "prettified" 
format without changing the contents, run

python path/to/raven/scripts/conversionScripts/standard.py test.xml

 - talbpaul, 2016-02-08
