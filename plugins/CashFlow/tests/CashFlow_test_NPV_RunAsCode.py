
import os
import sys
# run the Cash Flow plugin as stand alone code
os.system('python  ../src/CashFlow_ExtMod.py -iXML Cash_Flow_input_NPV.xml -iINP VarInp.txt -o out.out')

# read out.out and compare with gold
with open("out.out") as out:
  for l in out:
    pass

gold = float(l)
if (gold - 630614140.519) < 0.01:
  sys.exit(0)
else:
  sys.exit(1)
