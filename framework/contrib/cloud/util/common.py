"""
This module holds convenience functions for accessing information 

Copyright (c) 2013 `PiCloud, Inc. <http://www.picloud.com>`_.  All rights reserved.

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

import datetime
import logging
import os
import platform
import re
import sys
import time

from subprocess import Popen, PIPE, STDOUT

try:
    import json
except:
    # Python 2.5 compatibility
    import simplejson as json

import cloud
from . import credentials
from ..cloudlog import stdout as print_stdout

cloudLog = logging.getLogger('util.common')


########## Utilities for sending post request ############
def _send_request(request_url, data, jsonize_values=True):
    """Makes a cloud request and returns the results.
    
    * request_url: whee the request should be sent
    * data: dictionary of post values relevant to the request
    * jsonize_values: if True (default), then the values of the *data*
        dictionary are jsonized before request is made."""
    if jsonize_values:
        data = _jsonize_values(data)
    conn = cloud._getcloudnetconnection()
    return conn.send_request(request_url, data)

def _jsonize_values(dct):
    """Jsonizes the values of the dictionary, but not the keys."""
    jsonized = [(key, json.dumps(value)) for key, value in dct.items()]
    jsonized.append(('_jsonized', True))
    return dict(jsonized)

def _fix_time_element(dct, keys):
    if not hasattr(keys, '__iter__'):
        keys = [keys]

    FORMAT = '%Y-%m-%d %H:%M:%S'
    for key in keys:
        if dct.has_key(key):
            val = dct.get(key)
            dct[key] = (None if (val == 'None')
                        else datetime.datetime.strptime(val, FORMAT))

    return dct


########## Platform dependencies ###########
plat = platform.system()

def _rsync_path():
    base_dir = os.path.join(sys.prefix, 'extras') if plat == 'Windows' else ''
    return os.path.join(base_dir, 'rsync')

def _ssh_path():
    base_dir = os.path.join(sys.prefix, 'extras') if plat == 'Windows' else ''
    return os.path.join(base_dir, 'ssh')

def _check_rsync_dependencies():
    """Checks dependencies by trying the commands."""
    msg_template = 'Dependency %s not found'
    if plat == 'Windows':
        msg_template += ' in cloud installation'

    # check rsync exists
    status, _, _ = exec_command('%s --version' % _rsync_path(), pipe_output=True)
    if status:
        raise cloud.CloudException(msg_template % 'rsync')
    
    # check ssh exists
    status, _, _ = exec_command('%s -V' % _ssh_path(), pipe_output=True)
    if status:
        raise cloud.CloudException(msg_template % 'ssh')

class WindowsPath(object):
    """Processes Windows paths for easy conversion into cwRsync compatible
    format.  This involves replacing drive specification into
    /cygdrive/[drive_lettter] format, and replacing backward slashes with
    forward slashes.  While backward slashes will allow correctly locating the
    filesystem resource locally, it most likely will result in incorrect
    destinations on the remote side, as backward slashes will be interpreted as
    part of the name itself.
    """
    def __init__(self, path):
        """Path is any (absolute or relative) Windows path."""
        if os.path.isabs(path):
            drive, tail = os.path.splitdrive(path)
            self.drive = drive.rstrip(':')
            self.tail = tail.lstrip('\\').lstrip('/')
        else:
            self.drive = ''
            self.tail = path.lstrip('\\').lstrip('/')

    def __repr__(self):
        """This is where the rsync safe conversion takes place.  If there is a
        space in the path, encapsulates the path with quotes to assure the path
        will not be misinterpreted as being two or more paths.
        """
        drive = os.path.join('\\cygdrive', self.drive) if self.drive else ''
        path = os.path.join(drive, self.tail)
        path = path.replace('\\', '/')
        if " " in path:
            path = '"%s"' % path
        return path

def fix_path(path):
    return str(WindowsPath(path)) if plat == 'Windows' else path

_remote_path_delimiter = ':'

def is_local_path(path):
    return ((_remote_path_delimiter not in path) or
            (plat == 'Windows' and re.match(r'[a-zA-Z]:(\\|/)', path)))

def parse_local_paths(local_paths):
    """Validate local paths."""
    if not isinstance(local_paths, (tuple, list)):
        local_paths = [local_paths]
    
    parsed_paths = []
    for path in local_paths:
        path = fix_path(path)
        if path.count(_remote_path_delimiter):
            raise cloud.CloudException('Local path cannot contain "%s"' %
                                       _remote_path_delimiter)
        parsed_paths.append(path)
    return parsed_paths

def parse_remote_paths(remote_paths):
    """Validate remote paths."""
    if not isinstance(remote_paths, (tuple, list)):
        remote_paths = [remote_paths]
    
    resource_name = None
    parsed_paths = []
    for path in remote_paths:
        if path.count(_remote_path_delimiter) != 1:
            raise cloud.CloudException('Remote path must be resource_name:path')
        r_name, r_path = path.split(_remote_path_delimiter)
        resource_name = resource_name or r_name
        if not r_name:
            raise cloud.CloudException('Remote path must start with a resource name')
        if resource_name != r_name:
            raise cloud.CloudException('All remote paths must use the same resource')
        parsed_paths.append(r_path)
    return (resource_name, parsed_paths)


########## OS interaction ###########
def exec_command(cmd, pipe_output=True, redirect_stderr=False):
    """Simple shell exec wrapper, with no stdin."""
    out_pipe = PIPE if pipe_output else None
    err_pipe = STDOUT if redirect_stderr else out_pipe
    p = Popen(cmd, shell=True, stdout=out_pipe, stderr=err_pipe)
    stdout, stderr = p.communicate()
    status = p.returncode
    return (status, stdout, stderr)

def ssh_session(username, hostname, key_path, port=None, run_cmd=None):
    default_options = ['-o', 'StrictHostKeyChecking=no',
                       '-o', 'ConnectTimeout=60',
                       '-o', 'LogLevel=QUIET',
                       ]
    ssh_path = _ssh_path()
    cmd = [ssh_path]
    if port:
        cmd.extend(['-p', str(port)])
    cmd.extend(default_options)
    cmd.extend(['-i', fix_path(key_path)])
    cmd.append('%s@%s' % (username, hostname))
    if run_cmd:
        if hasattr(run_cmd, '__iter__'):
            run_cmd = ' '.join(run_cmd)
        cmd.append("'%s'" % run_cmd)
        pipe_output = True
    else:
        pipe_output = False
    cmd_str = ' '.join(cmd)
    return exec_command(cmd_str, pipe_output=pipe_output, redirect_stderr=True)

def rsync_session(src_arg, dest_arg, delete=False, pipe_output=True):
    """Perform an rsync operation over ssh using api_key's ssh key."""
    _check_rsync_dependencies()
    
    api_key = cloud.connection_info().get('api_key')
    key_file = credentials.get_sshkey_path(api_key)
    if not os.path.exists(key_file):
        raise cloud.CloudException('%s not present. Something must have errored '
                                   'earlier. See cloud.log' % key_file)
    key_file = str(WindowsPath(key_file)) if plat == 'Windows' else key_file

    ssh_shell = ' '.join([_ssh_path(), '-q',
                          '-o UserKnownHostsFile=/dev/null',
                          '-o StrictHostKeyChecking=no',
                          '-o LogLevel=QUIET',
                          '-i %s' % key_file])
    rsync_cmd = [_rsync_path(), '-avz']
    if plat == 'Windows':
        rsync_cmd.append('--chmod=u+rwx')
    if delete:
        rsync_cmd.append('--delete')
    rsync_cmd.extend(['-e "%s"' % ssh_shell, src_arg, dest_arg])
    rsync_cmd_str = ' '.join(rsync_cmd)
    cloudLog.debug('Performing rsync command: %s', rsync_cmd_str)
    print_stdout('Performing rsync...')
    return exec_command(rsync_cmd_str, pipe_output)

def ssh_server_job(keep_alive=False):
    """A job that waits until all ssh connections have terminated to close
    Set *keep_alive* to True to keep job running forever
    """
    poll_time = 4.0
    
    activated = False
    
    if keep_alive: # sleep forever
        time.sleep(315360000)
        return 

    while True:
        time.sleep(poll_time)
        p = Popen('ps -C sshd -o cmd | grep pts', shell=True, stdout=PIPE, stderr=PIPE)
        stdout, _ = p.communicate()
        lines = stdout.splitlines()
        
        if not lines:
            if activated:
                print 'No more SSH connections. Terminating job'
                break
        
        else:
            activated=True
        
    