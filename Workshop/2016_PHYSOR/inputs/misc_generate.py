#!/bin/env python3

number = 100

csvFile = open("dataSet.csv","w")
csvFile.write("n,filename\n")
for i in range(number):
  print(i,"dataSet_"+str(i)+".csv",sep=",",file=csvFile)
csvFile.close()

xmlFile = open("dataSet.xml","w")
xmlFile.write("""<data name="cluster_data" type="HistorySet">
<input>n</input>
<output>Time,x,y</output>
<inputFilename>dataSet.csv</inputFilename>
</data>
""")
xmlFile.close()
