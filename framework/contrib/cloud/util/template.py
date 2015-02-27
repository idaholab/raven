'''
Defines the logic necessary to handle templates and argument substitution for shell exec
'''
import re
from collections import defaultdict

variable_extract = re.compile(r'(?:[^\$\\]|\A){(\w+?)}')
def extract_vars(command_str):
    """Extract variables from a command string"""    
    matches = variable_extract.findall(command_str)
    return list(set(matches))

variable_extract_dup = re.compile(r'([^\$\\]|\A){{(\w+?)}}') # matches vars in duplicate curlies
def generate_command(command_str, var_dct, skip_validate = False):
    """Fill in variables in command_str with ones from var_dct"""
        
    if not skip_validate:
        validate_command_args(command_str, var_dct)
    
    # first duplicate all curlies    
    command_str = command_str.replace('{', '{{')
    command_str = command_str.replace('}', '}}')
    #print command_str
    
    # now un-duplicate template variables    
    command_str = variable_extract_dup.sub('\\1{\\2}', command_str)
    #print command_str
    
    formatted_cmd =  command_str.format(**var_dct)
    # replace escaped items
    formatted_cmd = formatted_cmd.replace('\\{', '{')
    formatted_cmd = formatted_cmd.replace('\\}', '}')
    return formatted_cmd


def validate_command_args(command_str, var_dct):
    command_vars = extract_vars(command_str)
    for var in command_vars:
        if var not in var_dct:
            raise ValueError('Paremeter %s in command "%s" was not bound' % (var, command_str))
    for var in var_dct:
        if var and var not in command_vars:
            raise ValueError('Argument named %s is not defined in command "%s"' % (var, command_str))

def _var_format_error(item):
    return ValueError('%s: Incorrect format. Variables must be formatted as name=value' % item)

def extract_args(arg_list, allow_map = False):
    """Returns dictionary mapping keyword to list of arguments.
    every list should be of length one if allow_map is false
    """
    kwds = {}
    if not arg_list:
        return kwds
    for arg in arg_list:
        
        parts = arg.split('=', 1)
        if len(parts) != 2:
            raise _var_format_error(arg)
        key, value = parts
        if not key or not value:
            raise _var_format_error(arg)
        if key in kwds:
            raise ValueError('key %s is multiply defined' % key)
        if not allow_map:
            kwds[key] = value
        else:
            kwd_values = []
            # split string on non-escaped ','
            buf_str = ''
            while True:
                idx = value.find(',')
                if idx == -1:
                    break
                if value[idx - 1] == '\\': #escaped
                    buf_str = buf_str + value[:idx+1]                    
                else:
                    kwd_values.append(buf_str + value[:idx])
                    buf_str = ''
                value = value[idx+1:]                
            if buf_str or value:
                kwd_values.append(buf_str + value)
            kwds[key] = kwd_values
    return kwds


    
        

if __name__ == '__main__':
    cmdstr = 'base'
    print generate_command(cmdstr, {})
    cmdstr = 'bash {} ${env} {1..2}'
    print generate_command(cmdstr, {})
    cmdstr = '{hello} bash {} ${{env_sub}} {1..2} {bye}'
    print generate_command(cmdstr, {'hello' : 'HELLO',
                                    'env_sub' : 'ENV_SUB',
                                    'bye' : 'BYE'})
    cmdstr = '{hello} bash {} ${{env_sub}} {1..2} \{bye\}'
    print generate_command(cmdstr, {'hello' : 'HELLO',
                                    'env_sub' : 'ENV_SUB'})            