"""
The package cloud.shortcuts is a convenience package that holds functions and classes 
that couple other cloud functions.

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

def get_connection_info(jid, port, protocol='tcp', timeout=None):
    """Given a ``jid`` and a ``protocol`` ``port``, the job is listening on, return a dictionary containing:
    
        * address: IP address of server job is running on  
        * port: External (translated) port that can be used to connect to the listening ``port``        
    
    For certain services, the returned dictionary will include other relevant keys:
    * username: username to use to connect to a service running on ``port`` (available with ssh (port 22))
    
    This function blocks until job has opened the port for listening. An optional
    ``timeout`` may be specified in seconds.
    
    Example::
     
        >>> get_connection_info(jid,'8000')  # request the translation for jid's TCP Port 8000        
        {'address': 'ec2-23-22-210-36.compute-1.amazonaws.com',
         'port': '22000'}
     
    Note that this function is an alias cloud.wait_for.port(jid, port, protocol, timeout) 
    """
    from .. import wait_for
    return wait_for.port(jid, port, protocol, timeout)
    
