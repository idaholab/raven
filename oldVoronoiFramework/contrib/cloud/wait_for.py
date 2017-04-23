"""
This module provides functions to "wait" on various aspects of a job

All functions expect a single jid.
A timeout may also be specified.

Return value/exception handling is function specific
"""
""" 

Copyright (c) 2012 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

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

import time as _time

from .cloud import CloudException as _CloudException
from .cloud import CloudTimeoutError as _CloudTimeoutError
from . import _getcloud


def _wait_for_test(jid, test_func, timeout=None, timeout_msg='Job wait timed out'):
    """Keep testing job until test_func(jid) returns a true value
    Return return value of test_func"""    
    
    poll_interval = 1.0    
    abort_time = _time.time() + timeout if timeout else None  

    while True:
        retval = test_func(jid)
        if retval:
            return retval
        
        if abort_time and _time.time() > abort_time:
            raise _CloudTimeoutError(timeout_msg, jid=jid)
             
        _time.sleep(poll_interval)        

def _checkint(var, name=''):
    if not isinstance(var, (int, long)):
        raise TypeError('%s must be a single integer' % name)


# all possible status transitions
_status_transitions = {'waiting' : ['stalled', 'killed', 'queued'],
                       'queued' : ['killed', 'processing'],
                       'processing' : ['killed', 'error', 'done'],
                       'stalled' : [],
                       'killed' : [],
                       'error' : [],
                       'done' : []
                       }

# all possible future states (could autogenerate from above)
_possible_future_statuses = {'waiting' : ['stalled', 'killed', 'queued', 'processing', 'done', 'error'],
                             'queued' : ['killed', 'processing', 'done', 'error'],
                             'processing' : ['killed', 'error', 'done'],
                             'stalled' : [],
                             'killed' : [],
                             'error' : [],
                             'done' : []
                             }

def _status_test_wrapper(test_status):
    """wrapper function to conduct status tests"""
    return status_test

def status(jid, test_status, timeout=None):
    """Wait until job's status is ``test_status``
    Raise CloudException if no longer possible to reach status
    
    Returns job's current status (which will be equal to test_status)    
    """
    _checkint(jid, 'jid')
    def status_test(jid):
        cl = _getcloud()
        cur_status = cl.status(jid)
        if test_status == cur_status:
            return cur_status
        if test_status not in _possible_future_statuses[cur_status]:
            raise _CloudException('Job has status %s. Will never (again) be %s' % (cur_status, test_status), 
                                 jid=jid, status=cur_status) 
        return False   
    
    return _wait_for_test(jid, status_test, timeout=timeout, 
                          timeout_msg='Job did not reach status %s before timeout' % test_status)

    
def port(jid, port, protocol='tcp', timeout=None):
    """Wait until job has opened ``port`` (under protocol ``protocol``) 
    for listening.
    Returns port translation dictionary. 
    
    See docstring for :func:`cloud.shortcuts.get_connection_info` for description
    of returned dictionary
    """
    
    _checkint(jid, 'jid')
    _checkint(port, 'port')

    cl = _getcloud()
        
    processing_poll_interval = 1.0 # polling on status wait
    port_poll_interval = 0.7 # polling on port wait
    
    abort_time = _time.time() + timeout if timeout else None
    status = None

    while True:
        jid_info = cl.info(jid, ['ports','status'])[jid]
        status = jid_info['status']
        port_info = jid_info.get('ports')
        
        if status in _possible_future_statuses['processing']:
            raise _CloudException('Job is already finished with status %s' % status, 
                        jid=jid, status=status)
        elif not port_info:
            if cl.is_simulated() and status == 'processing':
                return {'address' : '127.0.0.1', 'port' : port}
            elif abort_time and _time.time() > abort_time:
                raise _CloudTimeoutError('Job did not start processing before timeout', 
                                    jid=jid, status=status)            
            _time.sleep(processing_poll_interval)
            continue
        
        port_proto_info = port_info[protocol]        
        if port not in port_proto_info:
            if abort_time and _time.time() > abort_time:
                raise _CloudTimeoutError('Job did not open port %s before timeout' % port, 
                                    jid=jid)
            _time.sleep(port_poll_interval)
            continue
        
        return port_proto_info[port]           
    