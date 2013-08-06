#!/usr/bin/env python
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

def get_component_dict(yaml_data):
    components_yaml = [x for x in yaml_data if x["name"].endswith("Components")][0]
    
    #print [x["name"].split("/")[-1] for x in components_yaml["subblocks"]]
    #print [x for x in components_yaml["subblocks"][1]["parameters"] if x["name"] == "controlled"]
    
    component_dict = {}
    for component_block in components_yaml["subblocks"]:
        join_non_nulls = lambda y:(" ".join([x for x in y if x])).split()
        controlled = [x["options"] for x in component_block["parameters"] if x["name"] == "controlled"]
        controlled = join_non_nulls(controlled)
        monitored = [x["options"] for x in component_block["parameters"] if x["name"] == "monitored"]
        monitored = join_non_nulls(monitored)
        operators = [x["options"] for x in component_block["parameters"] if x["name"] == "operators"]
        operators = join_non_nulls(operators)
        sub_dict = {}
        #component_set = set()
        if len(controlled) > 0:
            sub_dict["controlled"] = controlled
            #component_set.update(controlled)
        if len(monitored) > 0:
            sub_dict["monitored"] = monitored
            #component_set.update(monitored)
        if len(operators) > 0:
            sub_dict["operators"] = operators            
        name = component_block["name"].split("/")[-1]
        parameters_dict = {}
        for parameter in component_block["parameters"]:
          var = parameter["name"]
          cpp_type = parameter["cpp_type"]
          description = parameter["description"]
          parameters_dict[var] = {"cpp_type":cpp_type,"description":description}
          #property_type = {}
          #  descriptions = {}
          #  for var in component_set:
          #    var_list = [(x["cpp_type"],x["description"]) 
          #                for x in component_block["parameters"]
          #                if x["name"] == var]
          #    var_type = [x[0] for x in var_list]
          #    var_description = [x[1] for x in var_list]
          #    if len(var_type) != 1:                
          #      print("WARNING unexpected property type",name,var,var_type)
          #    else:
          #      property_type[var] = var_type[0]
          #    descriptions[var] = " ".join(var_description)
          #  #print(name,property_type)
          #  sub_dict["property_type"] = property_type
          #  sub_dict["descriptions"] = descriptions
        sub_dict["parameters"] = parameters_dict
        component_dict[name] = sub_dict
        #print component_block["name"],controlled,monitored
    return component_dict
