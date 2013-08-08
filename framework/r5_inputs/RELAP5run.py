import sys
import relapdata
import os 

infile=sys.argv[1]
outfile=sys.argv[1]+'.o'
restart=sys.argv[1]+'.r'
csvfile=sys.argv[2]+'.csv'

cmd_str = '/home/nieljw/bin/relap5_403.x -i '+infile+'  -o '+outfile+' -r '+restart
os.system(cmd_str)
output=relapdata.relapdata(outfile)
output.write_csv(csvfile)
