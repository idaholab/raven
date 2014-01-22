#!/usr/bin/env python
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
import os, sys, subprocess, re
import yaml
import yaml_component

pathname = os.path.abspath(os.path.dirname(sys.argv[0]))
if len(sys.argv) < 2:
    print(sys.argv[0],"input_file_name [output_file_name [python_control_name]]")
    sys.exit(-1)
if len(sys.argv) > 2:
    output_file = open(sys.argv[2],"w")
    if len(sys.argv) > 3:
        control_name = sys.argv[3]
    else:
        control_name = sys.argv[2][:-2]
    python_output = open(control_name+".py","w")
else:
    output_file = sys.stdout
    python_output = sys.stdout
    control_name = ""

input_file = sys.argv[1]

MOOSE_DIR = pathname  + '/../../moose'
if  "MOOSE_DIR" in os.environ:
  MOOSE_DIR = os.environ['MOOSE_DIR']
elif "MOOSE_DEV" in os.environ:
  MOOSE_DIR = pathname + '/../devel/moose'


sys.path.append(MOOSE_DIR + '/scripts/common')

from ParseGetPot import readInputFile, GPNode

yaml_output = subprocess.check_output([pathname+"/../RAVEN-"+os.environ["METHOD"],"--yaml"])

yaml_part = yaml_output.split('**START YAML DATA**\n')[1].split('**END YAML DATA**')[0]

yaml_data = yaml.load(yaml_part)
#print [x["name"] for x in yaml_data]

controllable_output = subprocess.check_output([pathname+"/../RAVEN-"+os.environ["METHOD"],"-i",input_file,"--dump-ctrl"],stderr=subprocess.STDOUT)

controllable_part = controllable_output.split("== Controllable parameters ==")[1].split("=============================")[0]

def get_controllable_dict(controllable_data):
    controllable_dict = {}
    name = ""
    for line in controllable_data.split("\n"):
        if line.endswith(" [type]"):
            name = line[:-7]
            print("'"+name+"'")
            controllable_dict[name] = []
        elif line.startswith("  - "):
            parameter = line[4:]
            print("'"+name+"' '"+parameter+"'")
            c_list = controllable_dict.get(name,[])
            c_list.append(parameter)
            controllable_dict[name] = c_list
        elif line.strip() == '':
            #ignore
            name = ""
        else:
            print("ERROR expected parameter or name, got '"+line+"'")
    return controllable_dict

controllable_dict = get_controllable_dict(controllable_part)

print(controllable_dict)

input_data = readInputFile(input_file)

def print_gpnode(node, depth = 0, output = sys.stdout):
    indent = "  "
    prefix = indent*max(0,depth - 1)
    if depth == 0:
        pass
    elif depth == 1:
        output.write(prefix+"["+node.name+"]\n")
    else:
        output.write(prefix+"[./"+node.name+"]\n")
    
    for line in node.comments:
        output.write(prefix + indent + "# " + line+"\n")
    
    for param_name in node.params_list:
        param_value = node.params[param_name]
        if not re.match("^[0-9a-zA-Z.-]*$",param_value):
            #add quoting
            param_value = "'"+param_value+"'"
        param_comment = ""
        if param_name in node.param_comments:
            param_comment = " # "+node.param_comments[param_name]
        output.write(prefix + indent + param_name + " = " + param_value + param_comment+"\n")
    
    for child_name in node.children_list:
        print_gpnode(node.children[child_name],depth+1,output)
    
    if depth == 0:
        pass
    elif depth == 1:
        output.write(prefix+"[]\n")
    else:
        output.write(prefix+"[../]\n")

#print_gpnode(input_data)

component_node = input_data.children["Components"]

component_list = component_node.children_list

component_dict = yaml_component.get_component_dict(yaml_data)
print(component_dict)

def add_to_node(parent, node):
    parent.children_list.append(node.name)
    parent.children[node.name] = node

controlled_node = GPNode("Controlled",input_data)
if len(control_name) > 0:
    controlled_node.params_list = ["control_logic_input"]
    controlled_node.params["control_logic_input"] = control_name
add_to_node(input_data,controlled_node)
monitored_node = GPNode("Monitored",input_data)
add_to_node(input_data,monitored_node)

def split_parameter_name(name, hs_names = [], pipe_names = []):
    # Name can be something like inlet|outlet:K_reverse
    # which means that there is two possibilities inlet:K_reverse,outlet:K_reverse
    # There is a special name hs_name:foo, which means that all the heat structures need to be returned.
    # There is a special name pipe_name:foo, which means that all the pipe names need to 
    # be returned
    if name.startswith("hs_name:"):
        tail = name.split(":")[1]
        return [h+":"+tail for h in hs_names]
    if name.startswith("pipe_name:"):
        tail = name.split(":")[1]
        return [p+":"+tail for p in pipe_names]
    elif ":" in name:
        types, tail = name.split(":")
        types = types.split("|")
        return [t+"_"+tail for t in types]
    else:
        return [name]

monitored_names = []
controlled_names = []

for component_name in component_list:
    component_type = component_node.children[component_name].params["type"]
    name_of_hs = component_node.children[component_name].params.get("name_of_hs",None)
    if name_of_hs:
        name_of_hs = name_of_hs.split()
    else:
        name_of_hs = []
    inputs = component_node.children[component_name].params.get("inputs",None)
    outputs = component_node.children[component_name].params.get("outputs",None)
    pipe_names = []
    if inputs:
        #pipe_names += inputs.split()
        #Remove the (in) and (out), and split into list
        pipe_names += re.subn("\([a-z]*\)","",inputs)[0].split()
    if outputs:
        #pipe_names += outputs.split()
        pipe_names += re.subn("\([a-z]*\)","",outputs)[0].split()
        
    type_dict = component_dict.get(component_type,{})
    for monitored_combo in type_dict.get("monitored",[]):
        for monitored in split_parameter_name(monitored_combo,name_of_hs,pipe_names):
            for operator in type_dict.get("operators",[]):
                name = re.subn("[ :()]","_",(component_name+"_"+monitored+"_"+operator))[0]
                monitored_names.append(name)
                monitored_var_node = GPNode(name,monitored_node)
                monitored_var_node.params_list = ["component_name","path",
                                                  "operator","data_type"]
                monitored_var_node.params["path"] = monitored
                monitored_var_node.params["component_name"] = component_name
                monitored_var_node.params["operator"] = operator
                monitored_var_node.params["data_type"] =  type_dict["parameters"].get(monitored,{}).get("cpp_type","double") #type_dict["property_type"].get(monitored,"double")
                if monitored != "VOID_FRACTION_HEM":
                  #XXX VOID_FRACTION_HEM is only available if model_type
                  # is EQ_MODEL_HEM, so never use VOID_FRACTION_HEM
                  add_to_node(monitored_node,monitored_var_node)
                print(name,"path = ",monitored)
    for controlled_combo in type_dict.get("controlled",[]):
        for controlled in split_parameter_name(controlled_combo,name_of_hs,pipe_names):
            name = re.subn("[ :()]","_",(component_name+"_"+controlled))[0]
            if controlled in controllable_dict[component_name]:
              data_type = type_dict["parameters"].get(controlled,{}).get("cpp_type","double") #type_dict["property_type"].get(controlled,"double")
              controlled_var_node = GPNode(name,controlled_node)
              controlled_var_node.params_list = ["component_name","property_name","data_type"]
              controlled_var_node.params["property_name"] = controlled
              controlled_var_node.params["component_name"] = component_name
              controlled_var_node.params["data_type"] = data_type
              if controlled_var_node.params["data_type"] in {"double", "int", "float", "bool"}:
                controlled_names.append((name,data_type))
                add_to_node(controlled_node,controlled_var_node)
              else:
                print("Unexpected data_type ",controlled_var_node.params)
              print(name,"component_name="+component_name,"property_name="+controlled)
            else:
              print("NOT controllable",name,"component_name="+component_name,"property_name="+controlled)

    print(component_name,component_type,type_dict,name_of_hs)
    

def change_executioner(data):
    data.children["Executioner"].params["type"] = "RavenExecutioner"

change_executioner(input_data)


print_gpnode(input_data,output=output_file)

python_output.write("""

def initial_function(monitored, controlled, auxiliary):
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    mult = 1.0
""")
for name,data_type in controlled_names:
  if data_type == "double":
    python_output.write("    controlled."+name+" = mult*controlled."+name+"\n")
  else:
    python_output.write("    controlled."+name+" = controlled."+name+"\n")
python_output.write("""
    return

def control_function(monitored, controlled, auxiliary):
    print("monitored",monitored,"controlled",controlled,"auxiliary",auxiliary)
    mult = 1.0
""")
for name,data_type in controlled_names:
  if data_type == "double":
    python_output.write("    controlled."+name+" = mult*controlled."+name+"\n")
  else:
    python_output.write("    controlled."+name+" = controlled."+name+"\n")
python_output.write("""    return

""")
