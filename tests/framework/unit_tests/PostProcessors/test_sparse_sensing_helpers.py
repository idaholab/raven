import pytest
import xml.etree.ElementTree as ET
from ravenframework.Models.PostProcessors.SparseSensing import SparseSensing

def _parse(xml_str):
    spec = SparseSensing.getInputSpecification()()
    spec.parseNode(ET.fromstring(xml_str))
    return spec

def test_pivotParameter_is_parsed():
    xml = """<PostProcessor name='pp' subType='SparseSensing'>
      <Goal subType='reconstruction'>
        <features>X,Y,T</features><target>T</target>
        <basis>SVD</basis><nModes>2</nModes><nSensors>2</nSensors>
        <optimizer>QR</optimizer>
        <pivotParameter>time</pivotParameter>
        <reshape>snapshot</reshape>
      </Goal>
    </PostProcessor>"""
    pp = SparseSensing()
    pp._handleInput(_parse(xml))
    assert pp.pivotParameter == 'time'
    assert pp.reshape == 'snapshot'

def test_reshape_defaults_to_snapshot():
    xml = """<PostProcessor name='pp' subType='SparseSensing'>
      <Goal subType='reconstruction'>
        <features>X,Y,T</features><target>T</target>
        <basis>SVD</basis><nModes>2</nModes><nSensors>2</nSensors>
        <optimizer>QR</optimizer>
      </Goal>
    </PostProcessor>"""
    pp = SparseSensing()
    pp._handleInput(_parse(xml))
    assert pp.reshape == 'snapshot'
    assert pp.pivotParameter is None
