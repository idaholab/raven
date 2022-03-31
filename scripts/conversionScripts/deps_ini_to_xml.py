
"""
  Converts old dependency files from .ini to .xml
"""

import os
import sys
import configparser
import xml.etree.ElementTree as ET

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ravenframework')))
from utils import xmlUtils

def readConfig(inFile):
  """
    Reads in ini file.
    @ In, inFile, str, path to file object
    @ Out, config, ConfigParser, config read in
  """
  config = configparser.ConfigParser(allow_no_value=True)
  config.read(inFile)
  return config

def convert(config):
  """
    Converts config to xml
    @ In, config, ConfigParser, old dependencies
    @ Out, xml, ET.Element, converted xml
  """
  xml = ET.Element('dependencies')

  addCore(config, xml)
  addForge(config, xml)
  addPip(config, xml)
  addOptional(config, xml)

  addWindows(config, xml)
  addMac(config, xml)
  addLinux(config, xml)

  addWindowsPip(config, xml)
  addMacPip(config, xml)
  addLinuxPip(config, xml)

  addPipInstall(config,xml)
  addSkipCheck(config,xml)

  #print('DEBUGG new xml:')
  #print(xmlUtils.prettify(xml))
  return xml

def addCore(config, xml):
  """
    Adds core entries to xml
    @ In, config, ConfigParser, original config
    @ In, xml, ET.Element, root of dependencies
    @ Out, None
  """
  if 'core' in config:
    node = ET.Element('main')
    xml.append(node)
    print('DEBUGG entries:', list(config['core'].items()))
    for lib, vrs in config['core'].items():
      new = ET.Element(lib)
      node.append(new)
      new.text = vrs

def addForge(config, xml):
  """
    Adds forge entries to xml
    @ In, config, ConfigParser, original config
    @ In, xml, ET.Element, root of dependencies
    @ Out, None
  """
  if 'forge' in config:
    main = xml.find('main')
    if main is None:
      xml.append(ET.Element('main'))
    for lib, vrs in config['forge'].items():
      new = ET.Element(lib)
      main.append(new)
      new.text = vrs
      new.attrib['source'] = 'forge'

def addPip(config, xml):
  """
    Adds pip entries to xml
    @ In, config, ConfigParser, original config
    @ In, xml, ET.Element, root of dependencies
    @ Out, None
  """
  if 'pip' in config:
    main = xml.find('main')
    if main is None:
      xml.append(ET.Element('main'))
    for lib, vrs in config['pip'].items():
      new = ET.Element(lib)
      main.append(new)
      new.text = vrs
      new.attrib['source'] = 'pip'

def addOptional(config, xml):
  """
    Adds optional entries to xml
    @ In, config, ConfigParser, original config
    @ In, xml, ET.Element, root of dependencies
    @ Out, None
  """
  if 'optional' in config:
    main = xml.find('main')
    if main is None:
      xml.append(ET.Element('main'))
    for lib, vrs in config['optional'].items():
      new = ET.Element(lib)
      main.append(new)
      new.text = vrs
      new.attrib['optional'] = 'True'

def addWindows(config, xml):
  """
    Adds windows entries to xml
    @ In, config, ConfigParser, original config
    @ In, xml, ET.Element, root of dependencies
    @ Out, None
  """
  if 'windows' in config:
    main = xml.find('main')
    if main is None:
      xml.append(ET.Element('main'))
    for lib, vrs in config['windows'].items():
      new = main.find(lib)
      if new is None:
        new = ET.Element(lib)
        main.append(new)
        new.text = vrs
        new.attrib['os'] = 'windows'
      else:
        new.attrib['os'] = new.attrib['os']+',windows'

def addMac(config, xml):
  """
    Adds mac entries to xml
    @ In, config, ConfigParser, original config
    @ In, xml, ET.Element, root of dependencies
    @ Out, None
  """
  if 'mac' in config:
    main = xml.find('main')
    if main is None:
      xml.append(ET.Element('main'))
    for lib, vrs in config['mac'].items():
      new = main.find(lib)
      if new is None:
        new = ET.Element(lib)
        main.append(new)
        new.text = vrs
        new.attrib['os'] = 'mac'
      else:
        new.attrib['os'] = new.attrib['os']+',mac'

def addLinux(config, xml):
  """
    Adds linux entries to xml
    @ In, config, ConfigParser, original config
    @ In, xml, ET.Element, root of dependencies
    @ Out, None
  """
  if 'linux' in config:
    main = xml.find('main')
    if main is None:
      xml.append(ET.Element('main'))
    for lib, vrs in config['linux'].items():
      new = main.find(lib)
      if new is None:
        new = ET.Element(lib)
        main.append(new)
        new.text = vrs
        new.attrib['os'] = 'linux'
      else:
        new.attrib['os'] = new.attrib['os']+',linux'

def addWindowsPip(config, xml):
  """
    Adds windows pip entries to xml
    @ In, config, ConfigParser, original config
    @ In, xml, ET.Element, root of dependencies
    @ Out, None
  """
  if 'windows-pip' in config:
    main = xml.find('main')
    if main is None:
      xml.append(ET.Element('main'))
    for lib, vrs in config['windows-pip'].items():
      new = main.find(lib)
      if new is None:
        new = ET.Element(lib)
        main.append(new)
        new.text = vrs
        new.attrib['source'] = 'pip'
        new.attrib['os'] = 'windows'
      else:
        new.attrib['os'] = new.attrib['os']+',windows'

def addMacPip(config, xml):
  """
    Adds mac pip entries to xml
    @ In, config, ConfigParser, original config
    @ In, xml, ET.Element, root of dependencies
    @ Out, None
  """
  if 'mac-pip' in config:
    main = xml.find('main')
    if main is None:
      xml.append(ET.Element('main'))
    for lib, vrs in config['mac-pip'].items():
      new = main.find(lib)
      if new is None:
        new = ET.Element(lib)
        main.append(new)
        new.text = vrs
        new.attrib['source'] = 'pip'
        new.attrib['os'] = 'mac'
      else:
        new.attrib['os'] = new.attrib['os']+',mac'

def addLinuxPip(config, xml):
  """
    Adds linux pip entries to xml
    @ In, config, ConfigParser, original config
    @ In, xml, ET.Element, root of dependencies
    @ Out, None
  """
  if 'linux-pip' in config:
    main = xml.find('main')
    if main is None:
      xml.append(ET.Element('main'))
    for lib, vrs in config['linux-pip'].items():
      new = main.find(lib)
      if new is None:
        new = ET.Element(lib)
        main.append(new)
        new.text = vrs
        new.attrib['source'] = 'pip'
        new.attrib['os'] = 'linux'
      else:
        new.attrib['os'] = new.attrib['os']+',linux'

def addPipInstall(config, xml):
  """
    Adds pip alternate install entries to xml
    @ In, config, ConfigParser, original config
    @ In, xml, ET.Element, root of dependencies
    @ Out, None
  """
  if 'pip-install' in config:
    alt = ET.Element('alternate')
    xml.append(alt)
    alt.attrib['name'] = 'pip'
    for lib, vrs in config['pip-install'].items():
      new = ET.Element(lib)
      alt.append(new)
      new.text = vrs

def addSkipCheck(config, xml):
  """
    Adds skip check flags
    @ In, config, ConfigParser, original config
    @ In, xml, ET.Element, root of dependencies
    @ Out, None
  """
  if 'skip-check' in config:
    for lib, vrs in config['skip-check'].items():
      for install in xml:
        for exists in install:
          if exists.tag == lib:
            exists.attrib['skip_check'] = 'True'
            break

def write(xml, origPath):
  """
    Write new dep file
    @ In, xml, ET.Element, filled XML with dependencies
    @ In, origPath, str, original file path
    @ Out, None
  """
  newPath = os.path.join(os.path.dirname(origPath), 'dependencies.xml')
  xmlUtils.toFile(newPath, xml)
  print('Wrote new "dependencies.xml" to {}'.format(newPath))



if __name__ == '__main__':
  usageMessage = 'converter usage: python deps_ini_to_xml.py /path/to/dependencies.ini'
  if len(sys.argv) != 2:
    print(usageMessage)
    sys.exit(1)
  oldPath = os.path.abspath(sys.argv[1])
  if not os.path.isfile(oldPath):
    print(usageMessage)
    print('ERROR: File not found:', oldPath)
    sys.exit(1)
  config = readConfig(oldPath)
  xml = convert(config)
  write(xml, oldPath)



