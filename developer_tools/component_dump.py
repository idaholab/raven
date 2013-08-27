#!/usr/bin/env python
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
import os, sys, subprocess, re
import yaml
import yaml_component
from xml.sax.saxutils import escape

if len(sys.argv) > 1:
  output_file = open(sys.argv[1],"w")
else:
  output_file = sys.stdout

self_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
yaml_output = subprocess.check_output([self_dir+"/../RAVEN-"+os.environ["METHOD"],"--yaml"])

yaml_part = yaml_output.split('**START YAML DATA**\n')[1].split('**END YAML DATA**')[0]

yaml_data = yaml.load(yaml_part)

component_dict = yaml_component.get_component_dict(yaml_data)
#print(component_dict)

print("<html><body>",file=output_file)
for key in component_dict:
  sub_dict = component_dict[key]
  print("<h2>"+key+"</h2>",file=output_file)
  print("<dl>",file=output_file)
  for part in ["controlled","monitored","operators"]:
    if part in sub_dict:
      print("<dt>"+part+"</dt><dd>"+" ".join(sub_dict[part])+"</dd>",file=output_file)
  print("</dl>",file=output_file)
  print("<h3>Parameters</h3>",file=output_file)
  print("<dl>",file=output_file)
  parameters = component_dict[key]["parameters"]
  for parameter in parameters:
    print("<dt>"+parameter+" ("+escape(parameters[parameter]["cpp_type"])+")</dt><dd>"+
          parameters[parameter]["description"]+"</dd>",
          file=output_file)    
  print("</dl>",file=output_file)
print("</body></html>",file=output_file)
