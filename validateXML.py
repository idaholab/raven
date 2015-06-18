from subprocess import call

################################################################################
print('****************************** xmllint validation test ******************************')
call(['xmllint','--schema','example.xsd','--noout','example.xml'])

################################################################################
# print('****************************** lxml validation test ******************************')
# from lxml import etree

# xsdFile = open('example.xsd','r')
# fileContents = xsdFile.read()
# xsdFile.close()

# xmlschema_doc = etree.parse(fileContents)
# xmlschema = etree.XMLSchema(xmlschema_doc)

# testFile = open('example.xml','r')
# fileContents = testFile.read()
# testFile.close()

# doc = etree.parse(fileContents)
# if xmlschema.validate(doc):
#   print('Success')
# else:
#   print('Faillure')

################################################################################
print('****************************** PyXB validation test ******************************')
import pyxb


call(['pyxbgen','-u','example.xsd','-m','example'])

import example

xml = open('example.xml').read()
try:
  order = example.CreateFromDocument(xml,location_base='example.xml')
  print('Success')
  print(order.xsdConstraintsOK())
except pyxb.ValidationError as e:
  print('Failure')
  print(e.details())
