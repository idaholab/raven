'''
Created on Feb 21, 2013

@author: alfoa
'''
import Datas
import xml.etree.ElementTree as ET

class filter:
  def __init__(self):
    self.name   = ''
    self.input  = []
    self.output = []
  def readXml(self,xmlNode):
      
    root = xmlNode.getroot()
    self.name = root.get("name")
    
    try:
      inputs = root.find("Input").text
      # token-ize the string
      self.inputs = inputs.split()
      
    except:
      raise IOError('not found element "Input" in filter' + self)
  
    try:
      outputs = root.find("Output").text
      # token-ize the string
      self.outputs = outputs.split()
      
    except:
      raise IOError('not found element "Output" in filter' + self)      
  

class TimeFilter(filter):
  def __init__(self):
    filter.__init__(self)
    