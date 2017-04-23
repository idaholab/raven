"""
Interface to publish functions to PiCloud, allowing them to be invoked via the REST API

Api keys must be configured before using any functions in this module.
(via cloudconf.py, cloud.setkey, or being on PiCloud server)
"""
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
import sys
import re
import __builtin__

from . import _getcloud, _getcloudnetconnection
from .util.zip_packer import Packer
from .util import getargspec

try:
    from json import dumps as json_serialize
    from json import loads as json_deserialize
except ImportError: #If python version < 2.6, we need to use simplejson
    from simplejson import dumps as json_serialize
    from simplejson import loads as json_deserialize

_publish_query = 'rest/register/'
_remove_query = 'rest/deregister/'
_list_query = 'rest/list/'
_info_query = 'rest/info/'
_invoke_query = 'rest/invoke/%s/'
_invoke_map_query = 'rest/invoke/%s/map/'

"""
This module utilizes the cloud object extensively
The functions can be viewed as instance methods of the Cloud (hence accessing of protected variables)
"""

   

def _low_level_publish(func, label, out_encoding, arg_encoding, params, func_desc):
    cloud = _getcloud()
    conn = _getcloudnetconnection()
        
    os_env_vars = params.pop('os_env_vars', None)
    sfunc, sarg, logprefix, logcnt = cloud.adapter.cloud_serialize(func, 
        params['fast_serialization'], 
        [], logprefix='rest.', os_env_vars=os_env_vars)

    #Below is derived from HttpAdapter.job_add
    conn._update_params(params)
    cloud.adapter.dep_snapshot() #let adapter make any needed calls for dep tracking
    data = Packer()
    data.add(sfunc)
    params['data'] = data.finish()
    params['label'] = label
    params['description'] = func_desc
    params['arg_encoding'] = arg_encoding
    params['out_encoding'] = out_encoding
    resp = conn.send_request(_publish_query, params)
    return resp

def publish(func, label, out_encoding='json', **kwargs):
    """       
    Publish *func* (a callable) to PiCloud so it can be invoked through the PiCloud REST API
    
    The published function will be managed in the future by a unique (URL encoded) *label*. 
    
    *out_encoding* specifies the format that the return value should be in when retrieving the result
    via the REST API. Valid values are "json" for a JSON-encoded object and "raw", where the return value
    must be an str (but can contain any characters).
    
    The return value is the URL which can be HTTP POSTed to to invoke *func*. 
    See http://docs.picloud.com/rest.html for information about PiCloud's REST API    
    
    Certain special *kwargs* associated with cloud.call can be attached to the periodic jobs: 
        
    * _cores:
        Set number of cores your job will utilize. See http://docs.picloud.com/primer.html#choose-a-core-type/
        In addition to having access to more CPU cores, the amount of RAM available will grow linearly.
        Possible values for ``_cores`` depend on what ``_type`` you choose:
        
        * c1: 1                    
        * c2: 1, 2, 4, 8
        * f2: 1, 2, 4, 8, 16                                    
        * m1: 1, 2, 4, 8
        * s1: 1        
    * _env:
        A string specifying a custom environment you wish to run your job within.
        See environments overview at http://docs.picloud.com/environment.html                
    * _fast_serialization:
        This keyword can be used to speed up serialization, at the cost of some functionality.
        This affects the serialization of the spawned jobs' return value.
        The stored function will always be serialized by the enhanced serializer, with debugging features.
        Possible values keyword are:
                    
        0. default -- use cloud module's enhanced serialization and debugging info            
        1. no debug -- Disable all debugging features for result            
        2. use cPickle -- Use python's fast serializer, possibly causing PicklingErrors
    * _max_runtime:
        Specify the maximum amount of time (in integer minutes) jobs can run. If the job runs beyond 
        this time, it will be killed.                    
    * _os_env_vars:
        List of operating system environment variables that should be copied to PiCloud from your system
        Alternatively a dictionary mapping the environment variables to the desired values.                                                                            
    * _priority: 
        A positive integer denoting the job's priority. PiCloud tries to run jobs 
        with lower priority numbers before jobs with higher priority numbers.            
    * _profile:
        Set this to True to enable profiling of your code. Profiling information is 
        valuable for debugging, but may slow down your jobs.
    * _restartable:
        In the very rare event of hardware failure, this flag indicates that a spawned 
        job can be restarted if the failure happened in the middle of it.
        By default, this is true. This should be unset if the function has external state
        (e.g. it modifies a database entry)
    * _type:
        Choose the type of core to use, specified as a string:
        
        * c1: 1 compute unit, 300 MB ram, low I/O (default)                    
        * c2: 2.5 compute units, 800 MB ram, medium I/O
        * f2: 5.5 compute units, 3.75 GB ram, high I/O, hyperthreaded core                                    
        * m1: 3.25 compute units, 8 GB ram, high I/O
        * s1: Up to 2 compute units (variable), 300 MB ram, low I/O, 1 IP per core
                       
        See http://www.picloud.com/pricing/ for pricing information
    * _vol:
        A string or list of strings specifying a volume(s) you wish your jobs to have access to.

    """
    
    if not callable(func):
        raise TypeError( 'cloud.rest.publish first argument (%s) is not callable'  % (str(func) ))        
    
    m = re.match(r'^[A-Z0-9a-z_+-.]+$', label)
    if not m:
        raise TypeError('Label can only consist of valid URI characters (alphanumeric or from set(_+-.$)')
        
    #ASCII label:
    try:
        label = label.decode('ascii').encode('ascii')
    except (UnicodeDecodeError, UnicodeEncodeError): #should not be possible
        raise TypeError('label must be an ASCII string')
    
    try:
        docstring = '' if (func.__doc__ is None) else func.__doc__
        func_desc = (docstring).encode('utf8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        raise TypeError('function docstring must be an UTF8 compatible unicode string')
    
    if not isinstance(out_encoding, str):
        raise TypeError('out_encoding must be an ASCII string')
    
    cloud = _getcloud()
    params = cloud._getJobParameters(func, kwargs, 
                                             ignore=['_label', '_depends_on', '_depends_on_errors'])
    
    #argument specification for error checking and visibility
    argspec = getargspec(func)    
    argspec_serialized = json_serialize(argspec,default=str)
    if len(argspec_serialized) >= 255: #won't fit in db - clear defaults
        argspec[4] = {}
        argspec_serialized = json_serialize(argspec,default=str)        
    params['argspec'] = argspec_serialized
    
    resp = _low_level_publish(func, label, out_encoding, 'raw', params, func_desc)
    
    return resp['uri']

register = publish 

def remove(label):
    """
    Remove a published function from PiCloud
    """

    #ASCII label:
    try:
        label = label.decode('ascii').encode('ascii')
    except (UnicodeDecodeError, UnicodeEncodeError):
        raise TypeError('label must be an ASCII string')
    
    conn = _getcloudnetconnection()
    
    conn.send_request(_remove_query, {'label': label})   
    
deregister = remove    
   
def list():
    """
    List labels of published functions
    """ 
    #note beware: List is defined as this function
    conn = _getcloudnetconnection()
    resp = conn.send_request(_list_query, {})
    return resp['labels']

def info(label):
    """
    Retrieve information about a published function specified by *label*
    """
    conn = _getcloudnetconnection()
    resp = conn.send_request(_info_query, {'label':label})
    del resp['data']
    
    return resp

def _invoke(label, argument_dict, is_map=False):
    """low-level invoking with map ability"""
    # TODO: We actually have to json_serialize this.. somehows
    
    conn = _getcloudnetconnection()
    
    map_limit = conn.map_job_limit
    map_count = 1
    
    extracted_argdict = {}
    for name, arglist in argument_dict.items():
        extracted = []
        
        if not hasattr(arglist, '__iter__'):
            raise TypeError('%s must map to an iterable' % name)
            
        argiter = iter(arglist)
        
        iterated = False
        for arg in argiter:
            iterated = True
            if not hasattr(arg, 'read'): # encode anything but file objects
                try:
                    arg = json_serialize(arg)
                except (TypeError, UnicodeDecodeError):
                    raise TypeError('%s is not json encodable' % name)
            extracted.append(arg)
        
        if not iterated:
            raise ValueError('%s cannot be bound to an empty list' % name)
        
        extracted_len = len(extracted)
        if map_count == 1:
            if not is_map and extracted_len > 1:
                raise ValueError('%s can only have 1 item when allow_map is False' % name)
            
            if extracted_len > map_limit:
                raise ValueError('%s has %s items. Maximum is %s', name, extracted_len, map_limit)        
                        
            map_count = extracted_len
        elif extracted_len > 1 and extracted_len != map_count:
            raise ValueError('%s has %s items. Expected %s to match with other arguments' % (name, extracted_len, map_count))        
    
        extracted_argdict[name] = extracted    
    
    if is_map:        
        resp = conn.send_request(_invoke_map_query % label, extracted_argdict)        
        jid_res = resp['jids']
        rstart, rend = jid_res.split('-')
        return xrange(int(rstart), int(rend))
            
        
    else:
        resp = conn.send_request(_invoke_query % label, extracted_argdict)        
        return resp['jid']
            

def invoke(label, **kwargs):    
    """
    Explicitly invoke rest function defined by *label* with *kwargs*
    If value of a kwarg is a File-like object, it will be transmitted as binary data
    Otherwise, if value is a python object, json encoded version of it will be transmitted
    
    Returns jid
    
    """
    listified_args = dict((name, [val]) for name, val in kwargs.items())
    return _invoke(label, listified_args, is_map = False)

def invoke_map(label, **kwargs):    
    """
    Explicitly invoke rest function defined by *label* with *kwargs*
    Emulates PiCloud rest semantics:
    
    args is a dictionary of arguments to map.
        key is argument name        
        value is a list of arguments. 1 argument maps to one map job 
        if value has len(1), value will apply to all map jobs 
        All values with len(>1) must have same length
        Values need to be either file-like objects or json-encodable python objects         
        
    Be aware that this is more limiting than regular cloud.map
    -No more than 500 jobs
    -All of args will be read into memory before transport
    You will need to manually "chunk" maps if your arguments do not fit the above constraints
    """ 
    return _invoke(label, kwargs, is_map = True)