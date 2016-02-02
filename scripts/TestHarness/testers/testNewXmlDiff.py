import xml.etree.ElementTree as ET
import XMLDiff

a = ET.parse('dummy.xml')  #this is the gold file
b = ET.parse('dummy2.xml') #this is the test file

same,note = XMLDiff.compare_elements_2(a.getroot(),b.getroot())

print 'Same?',same
print note
