"""
This module holds convenience functions for accessing ssh information 

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

from .. import _getcloud, _getcloudnetconnection

def get_ssh_info(jid, timeout=None):
    """Return dictionary describing how to access job's ssh port.
    
    Blocks until ssh service is ready. An optional ``timeout`` in seconds may
    be specified. Returned dictionary provides:
    * address: Address to connect to
    * port: Port to connect to
    * username: username to connect with
    * identity: Path of identity file to use 
    
    """
    from . import get_connection_info
    from ..util.credentials import get_sshkey_path
    
    conn_dct = get_connection_info(jid, 22, timeout=timeout)

    api_key = _getcloudnetconnection().api_key
    key_path = get_sshkey_path(api_key)
    conn_dct['identity'] = key_path
    return conn_dct
