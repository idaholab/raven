'''
For running commands in a shell environment on PiCloud.
Shell programs are executed through template substitution
'''
from __future__ import with_statement
from __future__ import absolute_import

"""
Copyright (c) 2011 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

email: contact@picloud.com

The cloud package is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This package is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this package; if not, see 
http://www.gnu.org/licenses/lgpl-2.1.html    
"""

import os
import random
import re
import sys
import tempfile

from subprocess import Popen, PIPE

try:
    from json import dumps as json_serialize
except ImportError: #If python version < 2.6, we need to use simplejson
    from simplejson import dumps as json_serialize

from .util import template
from .rest import _low_level_publish
from .cron import _low_level_register
from .cloud import CloudException
from . import _getcloud

def _get_cloud_and_params(command, kwargs, ignore = []):
    for kwd in kwargs: 
        if not kwd.startswith('_'):
            raise ValueError('wildcard kwargs must be cloud kwd')        
    
    cloud = _getcloud()
    cloud._checkOpen()
    
    params = cloud._getJobParameters(None, kwargs, ignore)
    params['func_name'] = command
    params['fast_serialization'] = 2 # guarenteed to pass
    params['language'] = 'shell'
    
    return cloud, params    

def execute(command, argdict, return_file=None, ignore_exit_status=False, cwd=None, **kwargs):
    """Execute (possibly) templated *command*. Returns Job IDentifier (jid)
    
    * argdict - Dictionary mapping template parameters to values
    * return_file: Contents of this file will be result of job.  result is stdout if not provided
    * ignore_exit_status: if true, a non-zero exit code will not result in job erroring     
    * cwd: Current working directory to execute command within
    * kwargs: See cloud.call underscored keyword arguments
    
    """
    
    template.validate_command_args(command, argdict)
    
    _handle_args_upload(argdict)
    
    cloud, params = _get_cloud_and_params(command, kwargs)
    
    jid = cloud.adapter.job_call(params, _wrap_execute_program(command, return_file, ignore_exit_status, cwd = cwd), 
                                 (), argdict)
    return jid  

def execute_map(command, common_argdict, map_argdict, return_file=None, 
                ignore_exit_status=False, cwd=None, **kwargs):
    """Execute templated command in parallel. Return list of Job Identifiers (jids). See cloud.map
        for more information about mapping.  Arguments to this are:
        
    * common_argdict - Dictionary mapping template parameters to values for ALL map jobs
    * map_argdict - Dictionary mapping template parameters to a list of values
        The nth mapjob will have its template parameter substituted by the nth value in the list
        Note that all elements of map_argdict.values() must have the same length;
        The number of mapjobs produced will be equal to that length
    * return_file: Contents of this file will be result of job.  result is stdout if not provided
    * ignore_exit_status: if true, a non-zero exit code will not result in job erroring   
    * cwd: Current working directory to execute command within
    * kwargs: See cloud.map underscored keyword arguments     
    """
    #print 'c/m', common_argdict, map_argdict
    
    combined_dct = {}
    combined_dct.update(common_argdict)
    combined_dct.update(map_argdict)    
    template.validate_command_args(command, combined_dct)
    
    _handle_args_upload(common_argdict)
       
    # Convert map_argdict into a dist of dicts
    num_args = None
    
    map_dct_iters = {}
    # Error handling
    
    for key, val_list in map_argdict.items():
        if not num_args:
            num_args = len(val_list)
        if not val_list:
            raise ValueError('Key %s must map to a non-empty argument list' % key)
        elif num_args != len(val_list):
            raise ValueError('Key %s had %s args. Expected %s to conform to other keys' % (key, len(val_list), num_args))
        map_dct_iters[key] = iter(val_list)
        
    map_template_lists = [] # will be list of template dictionaries    
    
    if not num_args:
        raise ValueError('At least one element must be provided in map_argdict')
    
    for _ in xrange(num_args):
        map_template = {}
        for key, dct_iter in map_dct_iters.items():
            nxtval = next(dct_iter)
            map_template[key] = nxtval        
        
        _handle_args_upload(map_template)        
        map_template_lists.append(map_template)
    
    cloud, params = _get_cloud_and_params(command, kwargs)
        
    jids = cloud.adapter.jobs_map(params, 
                                 _wrap_execute_program(command, return_file, 
                                                       ignore_exit_status, common_argdict, cwd=cwd),                                 
                                None, map_template_lists)
    return jids


def rest_publish(command, label, return_file=None, 
                ignore_exit_status=False, **kwargs):
    """Publish shell *command* to PiCloud so it can be invoked through the PiCloud REST API
    The published function will be managed in the future by a unique (URL encoded) *label*.
    Returns url of published function. See cloud.rest.publish
    
    See cloud.shell.execute for description other arguments    
    See cloud.rest.publish for description of **kwargs
    """

    if not label:
        raise ValueError('label must be provided')
    m = re.match(r'^[A-Z0-9a-z_+-.]+$', label)
    if not m:
        raise TypeError('Label can only consist of valid URI characters (alphanumeric or from set(_+-.$)')
    try:
        label = label.decode('ascii').encode('ascii')
    except (UnicodeDecodeError, UnicodeEncodeError): #should not be possible
        raise TypeError('label must be an ASCII string')
    
    cloud, params = _get_cloud_and_params(command, kwargs,
                                          ignore=['_label', '_depends_on', '_depends_on_errors'] )
    
    # shell argspecs are dictionaries
    cmd_params = template.extract_vars(command)
    argspec = {'prms' : cmd_params,
               'cmd' : command}
    argspec_serialized = json_serialize(argspec)
    if len(argspec_serialized) >= 255: #won't fit in db - clear command
        del argspec['command']
        argspec_serialized = json_serialize(argspec)
        if len(argspec_serialized) >= 255: #commands too large; cannot type check
            argspec_serialized = json_serialize({})
    params['argspec'] = argspec_serialized
    
    return _low_level_publish(_wrap_execute_program(command, return_file, ignore_exit_status), 
                       label, 'raw', 'actiondct',
                       params, func_desc='command invoked in shell')['uri']

def cron_register(command, label, schedule, return_file = None,
                  ignore_exit_status=False, **kwargs):
    """Register shell *command* to be run periodically on PiCloud according to *schedule*
    The cron can be managed in the future by the specified *label*.
    
    Flags only relevant if you call cloud.result() on the cron job:
    return_file: Contents of this file will be result of job created by REST invoke.  
        result is stdout if not provided
    ignore_exit_status: if true, a non-zero exit code will not result in job erroring
    """

    cloud, params = _get_cloud_and_params(command, kwargs,
                                          ignore=['_label', '_depends_on', '_depends_on_errors'] )
    func = _wrap_execute_program(command, return_file, ignore_exit_status)
    
    return _low_level_register(func, label, schedule, params)
    
"""execution logic"""
def _execute_shell_program(command, return_file, ignore_exit_status, template_args, cwd = None):
    """Executes a shell program on the cloud"""
    
    _handle_args_download(template_args, cwd)    
    templated_cmd = template.generate_command(command, template_args)
    
    if not return_file: # must save commands stdout to a file                
        stdout_handle = PIPE
    else:
        stdout_handle = sys.stdout
        
    # ensure /home/picloud/ is present if any python interpreter is launched
    env = os.environ 
    cur_path = env.get('PYTHONPATH','')
    if cur_path:
        cur_path = ':%s'  % cur_path
    env['PYTHONPATH'] = '/home/picloud/' + cur_path
    
    #p = Popen(templated_cmd, shell=True, stdout=stdout_handle, stderr=PIPE, cwd=cwd, env=env)
    # execute in context of BASH for environment variables
    p = Popen(["/bin/bash", "-ic", templated_cmd], stdout=stdout_handle, stderr=PIPE, cwd=cwd, env=env)
    
    if stdout_handle == PIPE:
        # attach tee to direct stdout to file
        return_file = tempfile.mktemp('shellcmd_stdout')
        tee_cmd = 'tee %s' % return_file
        p_out = p.stdout
        tout = Popen(tee_cmd, shell=True, stdin=p_out, stdout=sys.stdout, stderr=sys.stderr, cwd=cwd)
    else:
        tout = None
    
    # capture stderr for exceptions
    stderr_file = tempfile.mktemp('shellcmd_stderr')
    tee_cmd = 'tee %s' % stderr_file
    p_err = p.stderr
    terr = Popen(tee_cmd, shell=True, stdin=p_err, stdout=sys.stderr, stderr=sys.stderr, cwd=cwd)
    
    retcode = p.wait()
    # give tee time to flush stdout
    terr.wait()
    if tout:  
        tout.wait()    
    
    if retcode:
        msg = 'command terminated with nonzero return code %s' % retcode
        if ignore_exit_status:
            print >> sys.stderr, msg
        else:
            msg += '\nstderr follows:\n'
            with open(stderr_file) as ferr:
                # ensure don't exceed storage limits
                ferr.seek(0,2)
                ferr_size = ferr.tell()                
                ferr.seek(max(0,ferr_size - 15000000), 0) 
                msg += ferr.read() 
            raise CloudException(msg)
    
    if cwd and not cwd.endswith('/'):
        cwd = cwd + '/'
    return_path = cwd + return_file if cwd and not return_file.startswith('/') else return_file
    
    try:
        with open(return_path,'rb') as f: # If this raises an exception, return file could not be read
            retval = f.read()
    except (IOError, OSError), e:
        if len(e.args) == 2:
            e.args = (e.args[0], e.args[1] + '\nCannot read return file!')
        raise    
    
    if stdout_handle == PIPE: 
        os.remove(return_file)
    os.remove(stderr_file)
        
    return retval

def _execute_program_unwrapper(command, return_file, ignore_exit_status, wrapped_args, template_args, cwd = None):
    """unwraps closure generated in _wrap_execute_program._execute_program_unwrapper_closure"""
    args = template_args
    if wrapped_args:
        args.update(wrapped_args)
    return _execute_shell_program(command, return_file, ignore_exit_status, args, cwd)

def _wrap_execute_program(command, return_file, ignore_exit_status, wrapped_args=None, cwd = None):
    """Used to put common arguments inside the stored function itself
    
    close over these arguments
    At execution, template_args are merged with wrapped_args
    """
    def _execute_program_unwrapper_closure(**template_args):
        """
        minimal function to avoid opcode differences between python2.6 and python2.7
        Code of this function is stored in pickle object; _execute_program_unwrapper is a global environment reference
        """
        return _execute_program_unwrapper(command, return_file, ignore_exit_status, 
                                          wrapped_args, template_args, cwd)
    
    return _execute_program_unwrapper_closure 

"""helper functions"""
"""File uploading logic

There are some inefficiencies here.
    By closing the file data in a function, we lose the ability to stream it from disk 
    In practical usage, this probably won't matter and can always be changed later
        by using a rest invoke interface

"""

action_default = 'action_default'
action_upload = 'action_upload'

def _encode_upload_action(file_name):
    # upload file by binding it to a closure    
    
    f = open(file_name,'rb')
    contents = f.read()
    f.close()
    
    base_name = os.path.basename(file_name)
    
    return {'action' : action_upload,
            'filename' : base_name,
            'contents' : contents}
    
def _encode_default_action(arg):
    return {'action' : action_default,
            'value' : arg}    

def _handle_args_upload(arg_dct):
    """"
    arg_dct is a dictionary describing a job
    Each key is parameter that maps to its argument value
    
    If an argument is a file, it is automatically replaced by a function 
    that handles file unpacking
    """
    
    for param, arg in arg_dct.items():
        if arg.startswith('@'): # defines a file
            arg_dct[param] = _encode_upload_action(arg[1:])
        else:
            arg_dct[param] = _encode_default_action(arg)
    
    
"""downloading"""

def _decode_upload_action(action_dct, cwd):
    """place data in the current directory
    file name is name 
    if name already exists, append random integers to name until it doesn't
    """
    
    name = action_dct['filename']
    contents = action_dct['contents']
    
    cloud = _getcloud()
    if not cloud.running_on_cloud(): # simulation
        name = tempfile.mktemp(suffix=name)        
    
    started = False
    while os.path.exists(name):
        if not started:
            name+='-'
        started = True
        name += str(random.randint(0,9))
    
    if cwd and not cwd.endswith('/'):
        cwd = cwd + '/'
        
    fullpath = cwd + name if cwd else name 
    
    # Write user-uploaded file to local storage. (Can fail due to permission issues)
    # Be sure it has executable permissions on incase it is a shell script
    f = os.fdopen(os.open(fullpath,os.O_CREAT|os.O_RDWR,0777),'wb')
    f.write(contents)
    f.close()
    
    return name # use local name to fill in template

def _decode_default_action(action_dct, cwd):
    return action_dct['value']

def _handle_args_download(arg_dct, cwd):
    decode_map = {
                  action_upload : _decode_upload_action,
                  action_default : _decode_default_action 
                  }
    
    for param, action_dct in arg_dct.items():
        arg_dct[param] = decode_map[action_dct['action']](action_dct, cwd)
