"""
PiCloud realtime management.
This module allows the user to manage their realtime cores programmatically.
See http://docs.picloud.com/tech_overview.html#scheduling
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


import cloud
import types
from cloud.util import fix_time_element

import logging, datetime


_request_query = 'realtime/request/'
_release_query = 'realtime/release/'
_list_query = 'realtime/list/'
_change_max_duration_query = 'realtime/change_max_duration/'


"""
Real time requests management
"""
def list(request_id=""):
    """Returns a list of dictionaries describing realtime core requests.
    If *request_id* is specified, only show realtime core request with that request_id
    
    The keys within each returned dictionary are:
    
    * request_id: numeric ID associated with the request 
    * type: Type of computation resource this request grants
    * cores: Number of (type) cores this request grants
    * start_time: Time when real time request was satisfied; None if still pending"""
    
    if request_id != "":
        try:
            int(request_id)
        except ValueError:
            raise TypeError('Optional parameter to list_rt_cores must be a numeric request_id')
    
    conn = cloud._getcloudnetconnection()
    rt_list = conn.send_request(_list_query, {'rid': str(request_id)})
    return [fix_time_element(rt,'start_time') for rt in rt_list['requests']]

def request(type, cores, max_duration=None):
    """Request a number of *cores* of a certain compute resource *type*  
    Returns a dictionary describing the newly created realtime request, with the same format
    as the requests returned by list_rt_cores.
    If specified, request will terminate after being active for *max_duration* hours
    """
    
    if max_duration != None:
        if not isinstance(max_duration, (int, long)):
            raise TypeError('Optional parameter max_duration should be an integer value > 0')
        if max_duration <= 0:
            raise TypeError('Optional parameter max_duration should be an integer value > 0')
    
    conn = cloud._getcloudnetconnection()
    return fix_time_element(conn.send_request(_request_query, 
                                               {'cores': cores,
                                                'type' : type,
                                                'cap_duration': max_duration if max_duration else 0}), 
                             'start_time')

def release(request_id):
    """Release the realtime core request associated with *request_id*. 
    Request must have been satisfied to terminate."""
    
    try:
        int(request_id)
    except ValueError:
        raise TypeError('release_rt_cores requires a numeric request_id')
    
    conn = cloud._getcloudnetconnection()
    conn.send_request(_release_query, {'rid': str(request_id)})

def change_max_duration(request_id, new_max_duration=None):

    try:
        int(request_id)
    except ValueError:
        raise TypeError('release_rt_cores requires a numeric request_id')
    
    if new_max_duration != None:
        if not isinstance(new_max_duration, (int, long)):
            raise TypeError('Optional parameter max_duration should be an integer value > 0')
        if new_max_duration <= 0:
            raise TypeError('Optional parameter max_duration should be an integer value > 0')
    
    conn = cloud._getcloudnetconnection()
    
    conn.send_request(_change_max_duration_query, {'rid': str(request_id), 'cap_duration':new_max_duration})

