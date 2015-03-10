"""
PiCloud account management.
This module allows the user to manage account information programmatically.
Currently supports api key management
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

_key_list = 'key/list/'
_key_get = 'key/%s/'
_key_activate = 'key/%s/activate/'
_key_deactivate = 'key/%s/deactivate/'
_key_create = 'key/'

def list_keys(username, password, active_only=False):
    """Returns a list of all api keys. If *active_only* is True, only
    active keys are returned. *username* and *password* should be your
    PiCloud login information."""
    
    conn = cloud._getcloudnetconnection()
    resp = conn.send_request(_key_list,
                             {},
                             get_values={'active_only': active_only},
                             auth=(username, password))
    
    return resp['api_keys']

def get_key(username, password, api_key):
    """Returns information including api_secretkey, active status, and
    note for the specified *api_key*. *username* and *password* should
    be your PiCloud login information."""
    
    conn = cloud._getcloudnetconnection()
    resp = conn.send_request(_key_get % api_key,
                             {},
                             auth=(username, password))
    
    return resp['key']



def activate_key(username, password, api_key):
    """Activates the specified *api_key*. *username* and *password*
    should be your PiCloud login information."""
    
    conn = cloud._getcloudnetconnection()
    resp = conn.send_request(_key_activate % api_key,
                             {},
                             auth=(username, password))
    
    return True

def deactivate_key(username, password, api_key):
    """Deactivates the specified *api_key*. *username* and *password*
    should be your PiCloud login information."""
    
    conn = cloud._getcloudnetconnection()
    resp = conn.send_request(_key_deactivate % api_key,
                             {},
                             auth=(username, password))
    
    return True

def create_key(username, password):
    """Creates a new api_key. *username* and *password*
    should be your PiCloud login information."""
    
    conn = cloud._getcloudnetconnection()
    resp = conn.send_request(_key_create,
                             {},
                             auth=(username, password))
    
    return resp['key']


def get_key_by_key(api_key, api_secretkey):
    """
    Similar to *get_key*, but access information via api_key credentials
    (api_key and api_secretkey).
    """
    
    conn = cloud._getcloudnetconnection()
    resp = conn.send_request(_key_get % api_key,
                             {},
                             auth=(api_key, api_secretkey))
    
    return resp['key']